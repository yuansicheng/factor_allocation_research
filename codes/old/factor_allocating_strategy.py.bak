#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-02-05

import os
import sys
import argparse
import logging

from framework.strategy import Strategy
import statsmodels.formula.api as smf
from scipy.optimize import minimize
import numpy as np

import pandas as pd


class FactorAllocating(Strategy):
    def __init__(self, *args, strategy_args={}, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.strategy_args = strategy_args
        self.last_weights = None

        self.loadFactors(self.strategy_args['factor_path'])
        self.mode = self.strategy_args['mode']
        self.factors = ['GROWTH', 'REAL', 'MMT', 'VOL']
        # use pd.MultiIndex for 3-dims data
        self.taa_exposure = pd.DataFrame(columns=pd.MultiIndex.from_product([self.factors, self.asset_list]))
        

    def backtestOneDay(self):
        # You can use self.current_date, self.user_close, self.user_yield and self.user_raw_data
        # save target weights to self.weights
        # set orders to self.orders
        # 20220520 update: self.user_asset_ages were added to user-data

        self.on_sale_assets = [asset for asset in self.on_sale_assets if self.asset_ages[asset]>3*365]

        self.this_date = self.current_date
        asset_close_df = self.user_close.iloc[-3*self.constants['DAY_OF_YEAR']:][self.on_sale_assets]
        asset_daily_yield_df = self.user_yield.iloc[-3*self.constants['DAY_OF_YEAR']:][self.on_sale_assets]
        self.calExposure(asset_close_df, asset_daily_yield_df)

        if self.current_date in self.update_date:
            self.update(asset_close_df, asset_daily_yield_df)    
            return
        # if not self.last_weights is None and self.current_date in self.rebalance_date:
        #     self.rebalance()
        #     return

    def update(self, asset_close_df, asset_daily_yield_df):
        # fmp: factor mimicking portfolio
        

        self.fmp_weight = pd.DataFrame(columns=self.on_sale_assets)
        self.fmp_return = pd.Series(index=self.factors)

        if self.mode == 't':
            factor_exposure = self.taa_exposure.iloc[-1].unstack()
        if self.mode == 's':
            if self.taa_exposure.shape[0] <= self.constants['DAY_OF_YEAR']:
                factor_exposure = self.taa_exposure.mean().unstack()
            else:
                factor_exposure = self.taa_exposure.iloc[-self.constants['DAY_OF_YEAR']:].mean().unstack()

        # after unstack operation, each index is sorted alphabetically, so we have to recover the sequence to ensure the matrix calculation results
        factor_exposure = factor_exposure.loc[self.factors]
        factor_exposure = factor_exposure[self.on_sale_assets]
        
        self.factorMimickingPortfolio(asset_close_df, asset_daily_yield_df, exposure=factor_exposure)

        self.weights[self.on_sale_assets] = self.optimalFactorPortfolio(asset_daily_yield_df)

        self.last_weights = self.weights[:]

    def rebalance(self):
        self.weights = self.weights.loc[self.on_sale_assets]
        pass 

    def afterBacktest(self):
        pass 

    def loadFactors(self, factor_path):
        factor_files = os.listdir(factor_path)

        factor_df = pd.DataFrame()
        for f in factor_files:
            factor_name = f.split('.')[0]
            tmp = pd.read_excel(os.path.join(factor_path, f))
            tmp.index = pd.to_datetime(tmp['日期'])
            factor_df[factor_name] = tmp.iloc[:, -1]
        factor_df.columns = ['GROWTH', 'REAL']

        self.factor_df = factor_df

    def calExposure(self, asset_close_df, asset_daily_yield_df):
        # 2nd step
        for asset in self.on_sale_assets:
            factors = self.factor_df.copy().loc[asset_close_df.index]
            factors['MMT'] = (asset_close_df[asset]/asset_close_df[asset].shift(self.constants['DAY_OF_YEAR'])).iloc[-self.constants['DAY_OF_YEAR']:]
            factors['VOL'] = (100 * asset_daily_yield_df[asset]).rolling(2*self.constants['DAY_OF_YEAR']).std()
            factors = factors.iloc[-self.constants['DAY_OF_YEAR']:]
            
            for factor in factors.columns:
                # construct df for fit
                tmp = pd.DataFrame({'x':factors[factor], 'y':asset_close_df[asset].loc[factors.index]})
                self.taa_exposure.loc[self.this_date, (factor, asset)] = smf.ols('y ~ x', data = tmp).fit().params[1]

        tmp = self.taa_exposure.loc[self.this_date].unstack().T
        self.taa_exposure.loc[self.this_date] = ((tmp-tmp.mean()) / tmp.std()).T.stack()

    def factorMimickingPortfolio(self, asset_close_df, asset_daily_yield_df, exposure=None):
        # 3rd and 4th step
        asset_close_df = asset_close_df.iloc[-2*self.constants['DAY_OF_YEAR']:]
        asset_daily_yield_df = asset_daily_yield_df.iloc[-self.constants['DAY_OF_YEAR']:] - 1
        for i, factor in enumerate(exposure.index):
            B = np.matrix(exposure.iloc[i]).astype('float64').T
            cov = np.matrix(asset_daily_yield_df.cov()).astype('float64')

            if self.mode == 't':
                R = np.matrix(asset_close_df.iloc[-1] / asset_close_df.iloc[-self.constants['DAY_OF_YEAR']] - 1).astype('float64').T
            else:
                # Use the historical mean of returns
                R = np.matrix(asset_close_df.rolling(self.constants['DAY_OF_YEAR']).apply(lambda x: x.iloc[-1]/x.iloc[0] -1).dropna()).astype('float64').T

            port = np.linalg.inv(B.T@np.linalg.inv(cov)@B)@B.T@np.linalg.inv(cov)
            port_return = port@R

            port = np.squeeze(np.array(port))
            port_return = np.squeeze(np.array(port_return)).mean()

            self.fmp_weight.loc[factor] = port
            self.fmp_return[factor] = port_return

    def optimalFactorPortfolio(self, asset_daily_yield_df):
        # 5th step
        asset_daily_yield_df = asset_daily_yield_df.iloc[-self.constants['DAY_OF_YEAR']:] - 1
        factor_portfolio_yield = pd.DataFrame(columns=self.factors)
        for factor in self.factors:
            factor_portfolio_yield[factor] = (self.fmp_weight.loc[factor] * asset_daily_yield_df).sum(axis=1)
        
        cov = np.matrix(factor_portfolio_yield.cov()).astype('float64')
        w0 = [1/len(self.factors)] * len(self.factors)
        Q = np.matrix(self.fmp_return).astype('float64').T

        def factorPortfolioReturn(w,cov,Q): 
            w = np.matrix(w).astype('float64').T
            return -(w.T @ Q - 0.5 * w.T @ cov @ w)[0,0]
        
        #set constraints
        cons = [] 
        cons.append({'type': 'eq', 'fun': lambda w: -sum(w) + 1})
        if self.mode == 's':
            bounds = tuple([(0,1)] * len(self.factors))
        else:
            bounds = ()

        factor_weights = minimize(factorPortfolioReturn, w0, constraints=cons, args=(cov, Q), bounds=bounds,  method='SLSQP').x
        

        # 6th step
        asset_weights = (self.fmp_weight * np.array(factor_weights)[:,np.newaxis]).sum()
        cov = np.matrix(asset_daily_yield_df.cov()).astype('float64')
        alpha = cov @ np.matrix(asset_weights).T
        
        # 7th step
        w0 = [1/len(self.factors)] * len(self.on_sale_assets)
        def assetPortfolioReturn(w,cov, alpha): 
            w = np.matrix(w).astype('float64').T
            r = (w @ alpha.T - 0.5 * w.T @ cov @ w)[0,0] * 1e3
            return -r

        #set constraints
        cons = [] 
        # cons.append({'type': 'ineq', 'fun': lambda w: sum(w)})
        cons.append({'type': 'eq', 'fun': lambda w: -sum(w) + 1})
        bounds = tuple([(0,0.35)] * len(self.on_sale_assets))

        asset_weights = minimize(assetPortfolioReturn, w0, constraints=cons, args=(cov, alpha), bounds=bounds,  method='SLSQP').x

        return asset_weights


