#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author    :   yuansc
# @Contact   :   yuansicheng@ihep.ac.cn
# @Date      :   2022-12-08

import os, sys, logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

DAY_OF_YEAR = 252

class Strategy:
    def __init__(self) -> None:
        pass

    def run(self, id_date, close_data):
        raise NotImplementedError

class BacktestManager:
    def __init__(self, strategy, asset_close, **kwargs) -> None:
        self._initArgs()
        self._args.update(kwargs)
        for k,v in self._args.items():
            setattr(self, k, v)

        asset_close.index = pd.to_datetime(asset_close.index)
        self._asset_close = asset_close
        self._asset_return = asset_close.pct_change()
        self._strategy = strategy

        self._setLoopDate()

        self._value = self.init_cash
        self._cash = self._value
        self._historcial_position = []
        self._historcial_weights = {}
        self._historical_value = pd.Series(dtype=float)

        self._count_date = 0
        self._position_columns = ['nominal_position', 'actual_position', 'weight']

    def getHistoricalValue(self, scale=False):
        if scale:
            return self._historical_value / self._historical_value.iloc[0]
        return self._historical_value.copy()
        
    def _initArgs(self):
        # args
        self._args = {}
        self._args['date_range'] = (datetime(2015,1,1), datetime(2015,12,31))
        self._args['frequency'] = DAY_OF_YEAR // 12
        self._args['look_back'] = DAY_OF_YEAR
        self._args['init_cash'] = 1e4

    def _setLoopDate(self):
        # loop date
        tmp = self._asset_close.index[self.look_back:]
        self._loop_date = tmp[(tmp>=self.date_range[0]) & (tmp<=self.date_range[1])]
        self._id_date = None

    def _setUsefulAsset(self):
        self._useful_close = self._asset_close.loc[:self._id_date].iloc[-self.look_back:].dropna(axis=1)
        self._id_date_return = self._asset_return.loc[self._id_date, self._useful_close.columns]

    def _updateWeightAndValue(self):
        if not self._historcial_position:
            return
        self._value = self._cash + self._historcial_position[-1].actual_position.sum()
        # print(self._cash)
        self._historcial_position[-1].weight = self._historcial_position[-1].nominal_position / self._value

        

    def _updateDailyReturn(self):
        if not self._historcial_position:
            self._historcial_position.append(pd.DataFrame(columns=self._position_columns))
            return
        this_position = self._historcial_position[-1].reindex(self._id_date_return.index).fillna(0)
        this_position.nominal_position = this_position.nominal_position * (1+self._id_date_return)

        self._cash += (this_position.nominal_position.sum() - self._historcial_position[-1].nominal_position.sum())
        self._historcial_position.append(this_position)

        self._updateWeightAndValue()

    def _updateStrategyWeights(self):
        weights = self._historcial_weights[self._id_date]
        if not weights.any():
            print('BackettestManager: skip {} '.format(self._id_date))
            return
        self._historcial_position[-1].nominal_position = weights.fillna(0) * self._value

        delta_cash = self._historcial_position[-1].nominal_position.abs().sum() - self._historcial_position[-1].actual_position.sum()
        self._cash -= delta_cash
        self._historcial_position[-1].actual_position = self._historcial_position[-1].nominal_position.abs()

        self._updateWeightAndValue()

        

    

    def run(self):
        for id_date in tqdm(self._loop_date):
            self._id_date = id_date

            # 1.set useful asset
            self._setUsefulAsset()

            # 2.update daily return
            self._updateDailyReturn()

            # 3.if necessary, run strategy
            if not self._count_date % self.frequency:
                # print('run strategy')
                self._historcial_weights[id_date] = self._strategy.run(id_date, self._useful_close)

                # 4.update weights
                self._updateStrategyWeights()

            self._count_date += 1


                

            self._historical_value[id_date] = self._value



# # test
# import matplotlib.pyplot as plt
# this_path = os.path.dirname(__file__)
# asset_data = pd.read_csv(os.path.join(this_path, '../data/assets/asset_prices.csv'), index_col=0)

# # print(asset_data)

# class MyStrategy(Strategy):
#     def run(self, close_data):
#         return pd.Series([1/close_data.shape[1]]*close_data.shape[1], index=close_data.columns)

# my_backtest = BacktestManager(MyStrategy(), asset_data)
# my_backtest.run()

# my_backtest._historical_value.plot()
# plt.show()
