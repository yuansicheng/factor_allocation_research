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
from pypfopt.plotting import plot_efficient_frontier

from scipy.optimize import minimize

from datetime import datetime

this_path = os.path.dirname(__file__)
if this_path not in sys.path:
    sys.path.append(this_path)


def getFactorExposure(returns, factors):
    '''
    使用多元回归计算因子暴露
    输入：资产净值和因子值
    输出：资产在因子上的暴露（β）
    '''
    assert returns.shape[0] == factors.shape[0]
    return sm.OLS(returns, factors.astype(float)).fit().params.iloc[1:]

def getFatorExposureFrame(asset_returns, factors, norm=True):
    '''
    敞口矩阵
    norm：是否归一化
    '''
    asset_returns = (asset_returns - asset_returns.mean()) / asset_returns.std()
    factors = (factors - factors.mean()) / factors.std()
    # 因子数据添加常数项
    factors = sm.add_constant(factors)
    exposure_frame = pd.DataFrame(columns=factors.columns[1:])
    for asset in asset_returns.columns:
        exposure_frame.loc[asset] = getFactorExposure(asset_returns[asset], factors) 
    if norm:
        exposure_frame = (exposure_frame - exposure_frame.mean()) / exposure_frame.std()
    return exposure_frame

def getFactorMimickingPortfolio(asset_returns, factors, scale=None):
    exposure_df = getFatorExposureFrame(asset_returns, factors)
    # print(exposure_df)

    # P = [B' Σ**-1 B]**-1 B' Σ**-1
    # P: 因子模拟投资组合
    # B: 风险敞口矩阵
    # Σ: 资产协方差

    # 加入截距项
    exposure_df['intercept'] = 1
    B = np.matrix(exposure_df)
    sigma = np.matrix(pd.DataFrame(asset_returns).cov())
    sigma_inv = np.linalg.pinv(sigma)


    # print(B)
    # print(sigma)
    
    P = np.linalg.pinv(B.T @ sigma_inv @ B) @ B.T @ sigma_inv
    # print(P)
    P = pd.DataFrame(P.T[:,:-1], index=list(asset_returns.keys()), columns=list(factors.keys()))

    if scale:
        P = P / P.abs().sum() * scale
    
    return P

def optimalFactorPortfolio(fmp_return):
    cov = np.matrix(fmp_return.cov()).astype('float64')
    w0 = [1/fmp_return.shape[1]] * fmp_return.shape[1]
    Q = np.matrix((1+fmp_return).cumprod().iloc[-1]-1).astype('float64').T

    # print(cov, w0, Q)

    def factorPortfolioReturn(w,cov,Q): 
        w = np.matrix(w).astype('float64').T
        return -(w.T @ Q - 0.5 * w.T @ cov @ w)[0,0]
    
    #set constraints
    cons = [] 
    cons.append({'type': 'eq', 'fun': lambda w: -sum(w) + 1})
    bounds = tuple([(0,1)] * fmp_return.shape[1])

    factor_weights = minimize(factorPortfolioReturn, w0, constraints=cons, args=(cov, Q), bounds=bounds,  method='SLSQP').x
    return pd.DataFrame(factor_weights, index=fmp_return.columns).T

def getAssetExpectedReturns(fmp_weights, factor_weights, asset_returns):
    asset_weights = np.matrix(factor_weights).dot(np.matrix(fmp_weights).T)
    cov = np.matrix(asset_returns.cov()).astype('float64')
    alpha = cov @ asset_weights
    return pd.DataFrame(alpha,index=asset_returns.columns).T

# # test
# this_path = os.path.dirname(__file__)
# if this_path not in sys.path:
#     sys.path.append(this_path)
# from load_data import *

# # 读入资产数据
# asset_data = getAssetData()
# # 读入因子数据
# factor_data = getFactorData()

# id_date = datetime(2017, 8, 1)
# look_back = 252 * 3

# factor_returns = factor_data.reindex(asset_data.index).fillna(method='ffill').loc[:id_date].iloc[-look_back:]
# asset_returns = asset_data.loc[:id_date].iloc[-look_back:]

# exposure = getFatorExposureFrame(asset_returns.pct_change().fillna(0), factor_returns.pct_change().fillna(0))
# fmp = getFactorMimickingPortfolio(asset_returns.pct_change().fillna(0), factor_returns.pct_change().fillna(0), scale=1)

# print(exposure)
# print(fmp)


