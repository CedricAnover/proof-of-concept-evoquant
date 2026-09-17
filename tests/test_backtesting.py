import numpy as np

from evoquant.backtest_engine.utils import *
from evoquant.base import SeriesBool

# Only import evo_vbt_backtester if vectorbt is available
try:
    from evoquant.backtest_engine.evo_bt import evo_vbt_backtester

    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
import pandas as pd

from evoquant.backtest_engine.validation import *

# Optional vectorbt import for tests
try:
    import vectorbt as vbt
    from vectorbt.base.array_wrapper import ArrayWrapper

    VBT_INSTALLED = True
except ImportError:
    vbt = None
    ArrayWrapper = None
    VBT_INSTALLED = False

import random

# Create sample OHLCV data for testing
dates = pd.date_range(start="2023-01-01", periods=500, freq="h")
df_ohlcv = pd.DataFrame(
    {
        "Date": dates,
        "Open": np.random.uniform(100, 110, 500),
        "High": np.random.uniform(110, 115, 500),
        "Low": np.random.uniform(95, 100, 500),
        "Close": np.random.uniform(100, 110, 500),
        "Volume": np.random.randint(1000, 10000, 500).astype(float),
    }
)
df_ohlcv.set_index("Date", inplace=True)
ss1 = df_ohlcv["Close"] > df_ohlcv["Close"].shift(5)
ss2 = df_ohlcv["Close"] > df_ohlcv["Close"].rolling(200).mean()
ss_bool = SeriesBool(ss1 | ss2)

# start_time = time.time()
# bt = Backtest(df_ohlcv, EvoStrategy, cash=10000., commission=.002, margin=1., trade_on_close=False, hedging=False, exclusive_orders=False)
# strat_params = dict(ser_bool=ss_bool, direction="LongShort", exit_encoded_entry=True, exit_after_n_bars=None, exit_after_n_days=None, start_date=None)
# res = bt.run(**strat_params)
# bt.plot()
# end_time = time.time()
# print("Elapsed time:", end_time - start_time, "seconds")
#
# print(res)
# print(res._trades.columns)
# print(res._trades.dtypes)
# print(res._trades)
# print(res._equity_curve)
# print(res._strategy)
#
# start_time = time.time()
# res = backtesting_result_cleaner(res._trades)
# end_time = time.time()
# print("Elapsed time for backtesting_result_cleaner:", end_time - start_time, "seconds")
# print("\n", res)
#
# IS, OOS = linear_is_oos(df_ohlcv, test_size=0.4)
# print("In-Sample:", IS)
# print("Out-of-Sample:", OOS)
# print()
# print(multi_linear_is_oos(df_ohlcv, n_splits=2, tain_ratio=.7))


# start_time = time.time()
# bt = Backtest(df_ohlcv, EvoStrategy, cash=10000., commission=.005, margin=1., trade_on_close=False, hedging=False, exclusive_orders=False)
# strat_params = dict(strategy_name="TestingTSL", ser_bool=ss_bool, direction="ShortOnly", exit_encoded_entry=True, stop_loss=(2, "Percent"), enable_tsl=True)
# res = bt.run(**strat_params)
# bt.plot()
# end_time = time.time()
# print("Backtesting.py Speed:", end_time - start_time, "seconds")

# Optional vectorbt-dependent code
if VBT_INSTALLED:
    from vectorbt.signals import nb

    test_arr = [random.choice([True, False]) for _ in range(df_ohlcv.shape[0])]
    test_arr = np.array(test_arr)

    # x_o, x_h, x_l, x_c, x_v, x_d

    def all_exits(from_i, to_i, col, x):
        # Array of Bools
        return np.array([from_i])  # Array of Indexes

    res = nb.generate_ex_nb(test_arr, 1, True, False, True, all_exits, df_ohlcv)

    print(res)
else:
    print("vectorbt not installed, skipping vectorbt-dependent tests")
