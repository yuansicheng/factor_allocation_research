#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author    :   yuansc
# @Contact   :   yuansicheng@ihep.ac.cn
# @Date      :   2022-10-23

import os, sys, logging

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

framework_path = os.path.join(os.path.dirname(__file__), '../../../../国君研究所/工作/FOF_portfolio_toolbox/framework')
if framework_path not in sys.path:
    sys.path.append(framework_path)

from alg.alg_base import AlgBase

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
        tmp = pd.DataFrame({'x':factor, 'y':returns})
        return smf.ols('y ~ x', data = tmp).fit().params[1]


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