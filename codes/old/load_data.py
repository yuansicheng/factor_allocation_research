#!/usr/bin/python
# -*- coding: utf-8 -*-

# @Author	:	yuansc
# @Contact	:	yuansicheng@ihep.ac.cn
# @Date		:	2022-11-15 

import os, sys, argparse, logging

import pandas as pd

framework_path = os.path.join(os.path.dirname(__file__), '../../../../国君研究所/工作/FOF_portfolio_toolbox/framework')
if framework_path not in sys.path:
    sys.path.append(framework_path)

this_path = os.path.dirname(__file__)

from component.asset.asset import Asset

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

    asset_dict = {}

    asset_data = pd.read_csv(asset_raw_data_file, index_col=0)
    for asset_name in asset_data.columns:
        raw_nav_data = asset_data[asset_name]

        # 创建资产实体并加入dict
        asset_obj = Asset(asset_name)
        asset_obj.setRawNavData(raw_nav_data)
        
        asset_dict[asset_name] = asset_obj
    return asset_dict 

def getFactorData(pca=False):
    factor_raw_data_path = os.path.join(this_path, '../data/factors')
    file_name = 'factors.csv'
    if pca:
        file_name = 'pca_factors.csv'
    factor_dict = {}
    factor_data = pd.read_csv(os.path.join(factor_raw_data_path, file_name), index_col=0)
    for factor_name in factor_data.columns:       
        # 创建因子实体并加入dict
        factor_obj = Asset(factor_name)
        factor_obj.setRawNavData(factor_data[factor_name])
        
        factor_dict[factor_name] = factor_obj

    return factor_dict