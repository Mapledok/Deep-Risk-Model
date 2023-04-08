"""
@Time : 2023/3/20 18:02
@Author : 十三
@Email : mapledok@outlook.com
@File : build_return_data.py
@Project : Deep Risk Model
"""
import torch
from operator import add
from functools import reduce
from itertools import chain
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm, trange


def build_data(time_start: str,
               time_stop: str,
               stock_size: int,
               num_feature: int) -> pd.DataFrame:
    dates = pd.date_range(start=time_start, end=time_stop, freq='B')
    stock_names = [f'stock{i}' for i in range(1, stock_size + 1)]
    features = np.random.randn(len(dates) * stock_size, num_feature)
    df = pd.DataFrame({'Date': np.repeat(dates, stock_size),
                       'Stock': np.tile(stock_names, len(dates)),
                       **{f'Feature{code}': features[:, code] for code in range(num_feature)}})
    return df


print(build_data(time_start='2023-01-02',
                 time_stop='2023-01-31',
                 stock_size=5,
                 num_feature=10))
