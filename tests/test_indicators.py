import numpy as np

from evoquant.base import *
from evoquant.indicators import *
from evoquant.indicators import COMPILE_RSI
from evoquant.signals import *

import pandas as pd

"""
Test indicators with synthetic data
"""

# Create synthetic OHLCV data for testing
np.random.seed(42)
n_rows = 1000
dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='h')
base_price = 50.0
df_ohlcv = pd.DataFrame({
    'Date': dates,
    'Open': base_price + np.cumsum(np.random.randn(n_rows) * 0.5),
    'High': base_price + np.cumsum(np.random.randn(n_rows) * 0.5) + np.abs(np.random.randn(n_rows) * 0.3),
    'Low': base_price + np.cumsum(np.random.randn(n_rows) * 0.5) - np.abs(np.random.randn(n_rows) * 0.3),
    'Close': base_price + np.cumsum(np.random.randn(n_rows) * 0.5),
    'Volume': np.random.randint(1000, 10000, n_rows).astype(float)
})
df_ohlcv.set_index('Date', inplace=True)
df_ohlcv['Volume'] = df_ohlcv['Volume'].astype(float)  # Convert Volume to Float and not int


def generate_random_bool_list(length):
    import random
    bool_list = [random.choice([True, False]) for _ in range(length)]
    return bool_list


test_ls_bool = generate_random_bool_list(8000)
test_ls_bool = np.array(test_ls_bool)
# print(type(test_ls_bool))

res = and3_or3(*[SeriesBool(np.array(generate_random_bool_list(8000))) for _ in range(12)]) # SeriesBool
print(res.to_pd_series())