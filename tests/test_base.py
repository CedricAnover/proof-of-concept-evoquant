import datetime
import random
import time

import numpy as np

from evoquant.base import *
from evoquant.indicators import *
from evoquant.signals import *

import pandas as pd

df_ohlcv = pd.read_csv(r"D:\Projects\FinDashAnalytics\Data\CTraderData\Clean\export-AUDUSD-Hour-BarChartHist.csv")
df_ohlcv['Volume'] = df_ohlcv['Volume'].astype(float) # Convert Volume to Float and not Integer
df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date']) # Datetime
df_ohlcv['StringColumn'] = pd.Series([random.choice(["Apple", "Orange", "Pear", "Mango"]) for _ in range(df_ohlcv.shape[0])], index=df_ohlcv.index)
df_ohlcv['StringColumn'] = df_ohlcv['StringColumn'].astype(str)
# print(df_ohlcv.dtypes)
# print(df_ohlcv)

x_date = SeriesDate(df_ohlcv['Date'])
x_open = SeriesOpen(df_ohlcv['Open'].values)
x_high = SeriesHigh(df_ohlcv['High'].values)
x_low = SeriesLow(df_ohlcv['Low'].values)
x_close = SeriesClose(df_ohlcv['Close'].values)
x_volume = SeriesVolume(df_ohlcv['Volume'].values)
x_ls = [x_open, x_high, x_low, x_close, x_volume]

from evoquant.signals import and_rule3


ser1 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))
ser2 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))
ser3 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))

start_time = time.time()

result = and_rule3(ser1, ser2, ser3)

result = result.series

end_time = time.time()
print("Backtesting.py Speed:", end_time - start_time, "seconds")


print(result)