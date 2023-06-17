import numpy as np

from evoquant.base import *
from evoquant.indicators import *
from evoquant.indicators import COMPILE_RSI
from evoquant.signals import *

import pandas as pd

"""
C:\ProgramData\Anaconda3\envs\ds_env\python.exe D:\Projects\FinDashAnalytics\PyScripts\evoquant\tests\test_main.py 
Iteration:  1
Expr Len:  3
Tree:  is_series_lower_shift_rule(Open, Lag(7))
<class 'AttributeError'>
'Terminal' object has no attribute 'to_pd_series'
"""

df_ohlcv = pd.read_csv("D:\Projects\FinDashAnalytics\Data\ASX OHLCV\HVN.csv")
df_ohlcv['Volume'] = df_ohlcv['Volume'].astype(float) # Convert Volume to Float and not int


def generate_random_bool_list(length):
    import random
    bool_list = [random.choice([True, False]) for _ in range(length)]
    return bool_list


test_ls_bool = generate_random_bool_list(8000)
test_ls_bool = np.array(test_ls_bool)
# print(type(test_ls_bool))

res = and3_or3(*[SeriesBool(np.array(generate_random_bool_list(8000))) for _ in range(12)]) # SeriesBool
print(res.to_pd_series())