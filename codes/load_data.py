#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-11-15 

import os, sys, argparse, logging
import pandas as pd

this_path = os.path.dirname(__file__)


def getAssetData(type_='asset'):
    # 读入资产原始数据
    asset_raw_data_path = os.path.join(this_path, '../data/assets')
        
    if type_ == 'fmp':
        file_name = 'fmp.csv'
    elif type_ == 'pca_fmp':
        file_name = 'pca_fmp.csv'
    else:
        file_name = 'asset_prices.csv'
    asset_raw_data_file = os.path.join(asset_raw_data_path, file_name)

    data = pd.read_csv(asset_raw_data_file, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data


def getFactorData(pca=False):
    factor_raw_data_path = os.path.join(this_path, '../data/factors')
    file_name = 'factors.csv'
    if pca:
        file_name = 'pca_factors.csv'

    data = pd.read_csv(os.path.join(factor_raw_data_path, file_name), index_col=0)
    data.index = pd.to_datetime(data.index)
    return data
