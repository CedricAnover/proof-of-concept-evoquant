import random
import time

import numpy as np
import pandas as pd

from evoquant.base import SeriesBool, SeriesClose, SeriesDate, SeriesHigh, SeriesLow, SeriesOpen, SeriesVolume
from evoquant.signals import and_rule3

# Create synthetic OHLCV data for testing
np.random.seed(42)
n_rows = 1000
dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="h")
base_price = 1.5
df_ohlcv = pd.DataFrame(
    {
        "Date": dates,
        "Open": base_price + np.cumsum(np.random.randn(n_rows) * 0.001),
        "High": base_price + np.cumsum(np.random.randn(n_rows) * 0.001) + np.abs(np.random.randn(n_rows) * 0.001),
        "Low": base_price + np.cumsum(np.random.randn(n_rows) * 0.001) - np.abs(np.random.randn(n_rows) * 0.001),
        "Close": base_price + np.cumsum(np.random.randn(n_rows) * 0.001),
        "Volume": np.random.randint(1000, 10000, n_rows).astype(float),
    }
)
df_ohlcv.set_index("Date", inplace=True)
df_ohlcv["Volume"] = df_ohlcv["Volume"].astype(float)  # Convert Volume to Float and not Integer
df_ohlcv["StringColumn"] = pd.Series(
    [random.choice(["Apple", "Orange", "Pear", "Mango"]) for _ in range(df_ohlcv.shape[0])], index=df_ohlcv.index
)
df_ohlcv["StringColumn"] = df_ohlcv["StringColumn"].astype(str)
# print(df_ohlcv.dtypes)
# print(df_ohlcv)

x_date = SeriesDate(df_ohlcv.index.to_series())
x_open = SeriesOpen(df_ohlcv["Open"].values)
x_high = SeriesHigh(df_ohlcv["High"].values)
x_low = SeriesLow(df_ohlcv["Low"].values)
x_close = SeriesClose(df_ohlcv["Close"].values)
x_volume = SeriesVolume(df_ohlcv["Volume"].values)
x_ls = [x_open, x_high, x_low, x_close, x_volume]

ser1 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))
ser2 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))
ser3 = SeriesBool(np.array([random.choice([True, False, np.nan]) for _ in range(1000000)]))

start_time = time.time()

result = and_rule3(ser1, ser2, ser3)

result = result.series

end_time = time.time()
print("Backtesting.py Speed:", end_time - start_time, "seconds")


print(result)
