#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author    :   yuansc
# @Contact   :   yuansicheng@ihep.ac.cn
# @Date      :   2022-10-23

import os, sys, logging

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from datetime import datetime

framework_path = os.path.join(os.path.dirname(__file__), '../../../../国君研究所/工作/FOF_portfolio_toolbox/framework')
if framework_path not in sys.path:
    sys.path.append(framework_path)

this_path = os.path.dirname(__file__)
if this_path not in sys.path:
    sys.path.append(this_path)

from alg.alg_base import AlgBase
from strategy.strategy_base import StrategyBase
from component.asset.asset import Asset
from component.position_manager.asset_position_manager import AssetPositionManager
from component.asset.group import Group
from component.position_manager.group_position_manager import GroupPositionManager

from import_func import getSvc
constant_svc = getSvc('ConstantSvc')
date_svc = getSvc('DateSvc')


class FactorExposureAlg(AlgBase):
    def __init__(self, name, args={}) -> None:
        super().__init__(name, args)

    def getFactorExposure(self, returns, factor):
        '''
        使用一元回归计算因子暴露
        输入：资产净值和因子值
        输出：资产在因子上的暴露（β）
        '''
        assert returns.shape[0] == factor.shape[0]
        # tmp = pd.DataFrame({'x':factor, 'y':returns})
        # return smf.ols('y ~ x', data = tmp).fit().params[1]

        factor_df = pd.DataFrame(factor)
        factor_df = sm.add_constant(factor_df)
        return sm.OLS(returns, factor_df.astype(float)).fit().params.iloc[1]


    def getFactorExposureV2(self, returns, factors):
        '''
        使用多元回归计算因子暴露
        输入：资产净值和因子值
        输出：资产在因子上的暴露（β）
        '''
        assert returns.shape[0] == factors.shape[0]
        return sm.OLS(returns, factors.astype(float)).fit().params.iloc[1:]


    def getFatorExposureFrame(self, asset_returns, factors):
        '''
        归一化敞口矩阵
        所有资产对某一因子的风险敞口均值为0，标准差为1
        '''
        exposure_frame = pd.DataFrame()
        for asset, returns in asset_returns.items():
            for factor, factor_values in factors.items():
                exposure_frame.loc[asset, factor] = self.getFactorExposure(returns, factor_values)

        exposure_frame = (exposure_frame - exposure_frame.mean()) / exposure_frame.std()
        return exposure_frame

    def getFatorExposureFrameV2(self, asset_returns, factors):
        '''
        归一化敞口矩阵
        所有资产对某一因子的风险敞口均值为0，标准差为1
        '''
        # 因子数据添加常数项
        factors_df = pd.DataFrame(factors)
        factors_df = sm.add_constant(factors_df)
        exposure_frame = pd.DataFrame(columns=factors_df.columns[1:])
        for asset, returns in asset_returns.items():
            exposure_frame.loc[asset] = self.getFactorExposureV2(returns, factors_df) 
        exposure_frame = (exposure_frame - exposure_frame.mean()) / exposure_frame.std()
        return exposure_frame
          

class FactorPortfolioAlg(AlgBase):
    def __init__(self, name, args={}) -> None:
        super().__init__(name, args)
        self._factor_exposure_alg = FactorExposureAlg('')

    def getFactorMimickingPortfolio(self, asset_returns, factors, scale=None, ols_mode='v2'):
        if ols_mode == 'v1':
            exposure_df = self._factor_exposure_alg.getFatorExposureFrame(asset_returns, factors)
        elif ols_mode == 'v2':
            exposure_df = self._factor_exposure_alg.getFatorExposureFrameV2(asset_returns, factors)

        # P = [B' Σ**-1 B]**-1 B' Σ**-1
        # P: 因子模拟投资组合
        # B: 风险敞口矩阵
        # Σ: 资产协方差

        B = np.matrix(exposure_df)
        sigma = np.matrix(pd.DataFrame(asset_returns).cov())
        sigma_rev = np.linalg.pinv(sigma)
        
        P = np.linalg.pinv(B.T @ sigma_rev @ B) @ B.T @ sigma_rev
        P = pd.DataFrame(P.T, index=list(asset_returns.keys()), columns=list(factors.keys()))

        if scale:
            P = P / P.abs().sum() * scale
        
        return P

class MvoAlg(AlgBase):
    def __init__(self, name) -> None:
        super().__init__(name)

    def run(self, data, weight_bounds=[-0.5, 0.5], total_weight=[-1, 1], target_return=None, target_risk=None, returns_data=False, mu=None, s=None, max_sharp=False, **kwargs):
        df = pd.DataFrame({k: v for k,v in data.items()}, dtype=float)
        
        mu = expected_returns.mean_historical_return(df, returns_data=returns_data) if mu is None else mu
        s = risk_models.sample_cov(df, returns_data=returns_data) if s is None else s
        logging.debug('mu = {}'.format(mu))
        logging.debug('s = {}'.format(s))
        ef = EfficientFrontier(mu, s, weight_bounds=weight_bounds, **kwargs)

        # if total_weight:
        #     sector_mapper = {a:'all' for a in data}
        #     ef.add_sector_constraints(sector_mapper, sector_lower={'all':total_weight[0]}, sector_upper={'all':total_weight[1]})

        # plot_efficient_frontier(ef)

        if max_sharp:
            _ = ef.max_sharpe()
        elif target_return:
            _ = ef.efficient_return(target_return)
        elif target_risk:
            _ = ef.efficient_risk(target_risk)
        return ef.clean_weights()


class FactorMimickingPortfolioStrategy(StrategyBase):
    def __init__(self, name, factor, args={}) -> None:
        self.indicator_period = constant_svc.DAY_OF_YEAR
        super().__init__(name, args)

        self._factor = getFactorData()[factor]


    def _initAlgDict(self):
        self._alg_dict['factor_portfolio_alg'] = FactorPortfolioAlg('_')

    # 设置数据集
    def _initDataset(self):
        # 父类方法初始化数据集并添加cash资产
        super()._initDataset(init_position_manager=True)

        asset_group = Group('assets')
        asset_group.setPositionManager(GroupPositionManager())
        self.getDataset().addChildGroup(asset_group)

        asset_dict = getAssetData()
        for asset_name, asset_obj in asset_dict.items():
            asset_group.addChildAsset(asset_obj)
            asset_obj.setPositionManager(AssetPositionManager())

    def run(self, id_date):
        # factor
        self._factor.setIdDate(id_date, self.indicator_period)
        factor_data = {self._factor.getName(): self._factor.getUsableNavData()}

        # asset
        self.setIdDate(id_date, self.indicator_period)

        asset_group = self.getDataset().getAllAsset('assets')
        asset_data = {asset_name: asset_obj.getUsableReturnData() for asset_name, asset_obj in asset_group.items() if asset_obj.getAge(id_date)>self.indicator_period}

        return self._alg_dict['factor_portfolio_alg'].getFactorMimickingPortfolio(asset_data, factor_data, scale=1).to_dict()[self._factor.getName()], []


        

    
def getAssetData():
    # 读入资产原始数据
    asset_raw_data_path = os.path.join(this_path, '../data/assets')
    asset_dict = {}
    for file_name in os.listdir(asset_raw_data_path):
        # 读入原始数据
        asset_name = file_name.split('.')[0]
        raw_data = pd.read_excel(os.path.join(asset_raw_data_path,file_name), index_col=0)
        raw_nav_data = raw_data['收盘价']
        raw_nav_data.index.name = 'date'
        
        # 创建资产实体并加入dict
        asset_obj = Asset(asset_name)
        asset_obj.setRawNavData(raw_nav_data)
        
        asset_dict[asset_name] = asset_obj
    return asset_dict 

def getFactorData():
    factor_raw_data_path = os.path.join(this_path, '../data/factors')
    factor_dict = {}
    for file_name in os.listdir(factor_raw_data_path):
        # 读入原始数据
        factor_name = file_name.split('.')[0]
        raw_data = pd.read_excel(os.path.join(factor_raw_data_path,file_name), index_col=0)
        raw_data.index.name = 'date'
        raw_data = raw_data.iloc[:, 0]

        # 去除第一行空值, 去除0值防止除0错误
        raw_data = raw_data.iloc[1:]
        raw_data.loc[raw_data==0] = 1e-6

        # logging.debug(raw_data)
        
        # 创建因子实体并加入dict
        factor_obj = Asset(factor_name)
        factor_obj.setRawNavData(raw_data)
        
        factor_dict[factor_name] = factor_obj

    return factor_dict





# # test
# raw_data_svc = getSvc('LxwWinddbRawDataSvc')
# date_svc.setTradeDays(raw_data_svc.getTradeDays())
# test_s = FactorMimickingPortfolioStrategy('fmps', 'growth')
# test_s.getDataset().print()


# id_date = datetime(2017, 8, 31)
# print(test_s.run(id_date))