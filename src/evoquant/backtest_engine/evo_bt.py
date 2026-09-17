"""
Assumptions:
- By default the backtest would use market order and not limit order.
- Exit At End of Week/Month/Year would require a variable because calculating it every backtest would be expensive.
- By default, long/short entry/exit signals are all executed on Open Prices.
- Some Time Exits, Stop Exits, and Volatility Exits are all executed on Close Prices, unfortunate in vbt.
- ExitAfterNBars and/or ExitAfterNDays are based on Open Prices. The rest is based on Close Prices (i.e. EndOfWMY).
    - ExitAfterNBars and/or ExitAfterNDays can be unioned with long/short exits.
- Volatility Signal based Exit can also be considered as part of long/short exits and executed on Open Prices.

Notes:
- Backtesting py implementation is slower but can be improved with python module multiprocessing using Pool.map method

References:
- https://stackoverflow.com/questions/53306927/chunksize-irrelevant-for-multiprocessing-pool-map-in-python
"""

# from evoquant.backtest_engine.utils import *
# from evoquant.backtest_engine.validation import *
# from evoquant.base import SeriesBool, SeriesDate

from .utils import *
from .validation import *
from evoquant import SeriesBool, SeriesDate

from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy, barssince

import vectorbt as vbt
from vectorbt.portfolio.enums import Direction, ConflictMode, DirectionConflictMode, OppositeEntryMode

from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import copy
import datetime

import operator
import quantstats, empyrical

from typing import Tuple, Union, Dict

class EvoStrategy(SignalStrategy):
    """
    EvoStrategy Class has to be modified in DEAP GP to pass on the SeriesBool Signals.
    EvoStrategy.ser_bool = s(:SeriesBool)

    Once it runned, you can extract the result objects from Backtest.run(). For example:
    bt = Backtest(df_ohlcv, Strategy, ...)
    res = bt.run()
    res._trades : DataFrame [Size,EntryBar,ExitBar,EntryPrice,ExitPrice,PnL,ReturnPct,EntryTime,ExitTime,Duration]
                            +[Direction,MAE,MFE]
                This DataFrame is to be cleaned with a cleaner function.
    res._equity_curve : Dataframe with Columns[$Equity, DrawdownPct%, DrawdownDuration] Equity starts from the Initial cash. To be converted to returns for performance calculations.
    res._strategy : Strategy Instance with its parameters
    """
    # We will use the vectorized approach, but edit the inside for extra control on exits and other behaviour.

    ser_bool: SeriesBool = None  # Throw error in init if not changed

    # The following properties cannot be modified once set by the user.
    strategy_name:str = None
    direction: str = None  # ["LongOnly", "ShortOnly", "LongShort"]

    start_date = None # (datetime) Start Date of the Backtest. Important for IS-OOS and Validation. Default=min(df_ohlcv.index)
    end_date = None  # (datetime) End Date of the Backtest. Important for IS-OOS and Validation. Default=max(df_ohlcv.index)

    exit_encoded_entry:bool = True

    exit_after_n_bars:int = None
    exit_after_n_days:int = None

    # TODO: Implement ExitAtEndOfWeek and ExitAtEndOfmonth
    exit_end_of_week:bool = False
    exit_end_of_month:bool = False

    exit_when_pnl_lessthan:float = None

    # For Fixed SL & TP
    # e.g. (1%->$(1/100)*EntryPrice +- EntryPrice, 'Percent') (2->$2*0.0001 +- EntryPrice, 'Pip', 0.0001) (2->$2*0.01, "Point", $0.01) ["Percent", "Pip", "Point"]
    stop_loss:tuple = tuple()

    # TODO: Implement EvoStrategy for Parameter take_profit
    take_profit:tuple = tuple()
    # Trade Size. If float between 0. & 1., then its it is interpreted as a fraction of current available liquidity.
    #             If int at least 1, indicates an absolute number of units.

    enable_tsl:bool = False

    trade_size:Union[float, str] = 0.99

    def init(self):
        super().init()
        # Edit the behaviour here (entries, exits, and others)...
        if (self.ser_bool == None) or (not isinstance(self.ser_bool, SeriesBool)):
            raise ValueError("EvoStrategy.ser_bool requires a SeriesBool instance.") # Throw error if self.ser_bool is not given
        if (self.direction not in ['LongOnly', 'ShortOnly', 'LongShort']): raise ValueError("Direction has to be given ['LongOnly', 'ShortOnly', 'LongShort'].")
        if (self.strategy_name is None or not isinstance(self.strategy_name, str)): raise ValueError("Strategy Name must be given")
        if self.enable_tsl: assert len(self.stop_loss) != 0, "Stop-Loss must be given for Trailing Stop"

        # self.ser_bool = self.ser_bool.to_pd_series(index=self.data.index) # Make sure SeriesBool is using datetime index

        # ---- Private Properties
        self._price_delta = None

        if self.start_date == None:
            self.start_date = self.data.index.min().date()
        if self.end_date == None:
            self.end_date = self.data.index.max().date()
        # print("Start: ", self.start_date)
        # print("End: ", self.end_date)
        # print(type(self.data))
        # Modify Data, based on the Start and End Date. May become useful for Validation, ISOOS, and WFA.

        # Calculate Signals as pd.Series with dtype=int (for Backtesting py)
        # Note that we still have to implement the exits in next() method.
        if self.direction == "LongOnly":
            signal = self.ser_bool.to_pd_series(index=self.data.index).astype(int).diff().fillna(0).replace(-1, 0) # Converting to Pulse for Long
            if self.exit_encoded_entry:
                self.long_exit = self.I(lambda: extract_start_end_true(self.ser_bool.to_pd_series(index=self.data.index).shift(1).fillna(False), start=False), plot=False)
            entry_size = signal * self.trade_size
        if self.direction == "ShortOnly":
            signal = self.ser_bool.to_pd_series(index=self.data.index).astype(int).diff().fillna(0).replace(-1, 0).mul(-1) # Converting to Pulse for Short
            if self.exit_encoded_entry:
                self.short_exit = self.I(lambda: extract_start_end_true(self.ser_bool.to_pd_series(index=self.data.index).shift(1).fillna(False), start=False), plot=False)
            entry_size = signal * self.trade_size
        if self.direction == "LongShort":
            # Same as "LongOnly", difference is that: 1) We dont replace -1 by 0, 2) In the Backtest parameters, we need to set exclusive_orders=True
            signal = self.ser_bool.to_pd_series(index=self.data.index).astype(int).diff().fillna(0)
            if self.exit_encoded_entry:
                self.long_exit = self.I(lambda: extract_start_end_true(self.ser_bool.to_pd_series(index=self.data.index), start=False).shift(1).fillna(False), plot=False)
                self.short_exit = self.I(lambda: extract_start_end_true(~self.ser_bool.to_pd_series(index=self.data.index), start=False).shift(1).fillna(False), plot=False)
            entry_size = signal * self.trade_size

        # Treat signal as an indicator, so we control the logic when there is conflict with entries.
        self.signal = self.I(lambda: signal, plot=False) # LongOnly{1,0}, ShortOnly{1,0}, LongShort{1,0,-1}

        # Set the signal for the Super Class SignalStrategy
        self.set_signal(entry_size=entry_size, exit_portion=None, plot=False)

    def next(self):
        # print("-"*200)
        # print(self.data.index[-1].date(), "|", f"Close={self.data.Close[-1]}", "|", f"#Trades={len(self.trades)}")

        # # Skip if We are out of Start & End Dates.
        # if self.data.index[-1].date() < self.start_date:
        #     return
        # if self.data.index[-1].date() > self.end_date:
        #     # If we already at the End Date, close all open positions.
        #     for trade in self.trades: trade.close()
        #     return
        
        # Run all Exits
        self.run_fixed_stop_loss() # Fixed Stop-Loss
        self.run_fixed_take_profit() # Fixed Take-Profit
        self.update_tsl() # Update TSL if enabled
        self.run_exit_encoded_entry() # Exit of Entry if Entry was a Filter Signal
        self.run_exit_after_n_bars() # Exit After N Bars
        self.run_exit_after_n_days() # Exit After N Days
        self.run_exit_when_pnl_lessthan() # Exit when trade position PnL is less than a certain $ value.

        self.run_entries() # Entries

    def run_entries(self):
        # Skip entry if we have open positions. But this will depend on the direction.
        if self.direction == "LongOnly":
            if len(self.trades) == 1: # Skip next, if we have 1 long position
                assert self.trades[0].is_long
                return
            elif len(self.trades) == 0:
                super().next()  # Super is SignalStrategy. This is for running the Entries.
        if self.direction == "ShortOnly":
            if len(self.trades) == 1:  # Skip next, if we have 1 short position
                assert self.trades[0].is_short
                return
            elif len(self.trades) == 0:
                super().next()  # Super is SignalStrategy. This is for running the Entries.
        if self.direction == "LongShort": # {1,0,-1}
            if len(self.trades) == 1:
                if self.trades[0].is_long:
                    if self.signal[-1] == 1:
                        return
                    elif self.signal[-1] == -1:
                        self.trades[0].close()
                        super().next()
                if self.trades[0].is_short:
                    if self.signal[-1] == -1:
                        return
                    elif self.signal[-1] == 1:
                        self.trades[0].close()
                        super().next()
            else:
                assert len(self.trades) == 0
                super().next()

    def run_exit_after_n_bars(self):
        if self.exit_after_n_bars == None:
            return
        if len(self.trades) == 0:
            # If we have no open trades, then pass.
            return
        assert len(self.trades) == 1
        trade = self.trades[0]
        if barssince(np.where(self.signal == -1, 1, self.signal).astype(bool)) >= self.exit_after_n_bars:
            trade.close()

    def run_exit_after_n_days(self):
        if self.exit_after_n_days == None:
            return
        if len(self.trades) == 0:
            # If we have no open trades, then pass.
            return
        assert len(self.trades) == 1
        trade = self.trades[0]

        if (self.data.index[-1].date() - trade.entry_time.date()).days >= self.exit_after_n_days:
            # print(f"{self.data.index[-1].date()} | We are closing a trade placed at {trade.entry_time.date()}. Time Delta is {(self.data.index[-1].date() - trade.entry_time.date()).days} days.")
            trade.close()

    def run_exit_encoded_entry(self):
        if not self.exit_encoded_entry:
            return

        if len(self.trades) == 0:
            # If we have no open trades, then pass.
            return

        assert len(self.trades) == 1
        trade = self.trades[0]
        if self.direction == "LongOnly":
            assert trade.is_long
            if self.long_exit[-1]:
                # print(f"Closed Long Position at {self.data.index[-1].date()}")
                trade.close()
        if self.direction == "ShortOnly":
            assert trade.is_short
            if self.short_exit[-1]:
                # print(f"Closed Short Position at {self.data.index[-1].date()}")
                trade.close()
        if self.direction == "LongShort":
            if self.long_exit[-1]:
                # print(f"Closed Long Position at {self.data.index[-1].date()}")
                trade.close()
            if self.short_exit[-1]:
                # print(f"Closed Short Position at {self.data.index[-1].date()}")
                trade.close()

    def run_fixed_stop_loss(self):
        if len(self.stop_loss) == 0:
            # If we didn't pass a stop-loss tuple parameter, then pass.
            return

        if len(self.trades) == 0:
            # If we have no open trades, then pass.
            return
        else:
            assert len(self.trades) == 1
            trade = self.trades[0] # Remember that Trade.sl & Trade.tp are prices.
            if trade.sl != None:
                # If open trade already has SL, then pass.
                return
            else:
                # Calculate the SL Prices in different units (Percent, Pips, Points)
                if self.stop_loss[1] == "Percent": # e.g. (1%, 'Percent')
                    price_delta = (self.stop_loss[0]/100.)*trade.entry_price
                elif self.stop_loss[1] == "Pip": # e.g. (2->$2*0.0001 +- EntryPrice, 'Pip', 0.0001)
                    price_delta = (self.stop_loss[0]*self.stop_loss[2])
                elif self.stop_loss[1] == "Point": # e.g. (2->$2*0.01, "Point", $0.01)
                    price_delta = self.stop_loss[0]*self.stop_loss[2]
                else:
                    raise
                # Store Price Delta in self._price_delta, so TSL can access it.
                self._price_delta = price_delta

                # Modify the Trade.sl property (trade.sl is None)
                if trade.is_long:
                    trade.sl = trade.entry_price - price_delta
                if trade.is_short:
                    trade.sl = trade.entry_price + price_delta
                return

    def run_fixed_take_profit(self):
        if len(self.take_profit) == 0:
            # If we didn't pass a take-profit tuple parameter, then pass.
            return

        if len(self.trades) == 0:
            # If we have no open trades, then pass.
            return
        else:
            assert len(self.trades) == 1
            trade = self.trades[0] # Remember that Trade.sl & Trade.tp are prices.
            if trade.tp != None:
                # If open trade already has TP, then pass.
                return
            else:
                # Calculate the TP Prices in different units (Percent, Pips, Points)
                if self.take_profit[1] == "Percent": # e.g. (1%, 'Percent')
                    price_delta = (self.take_profit[0]/100.)*trade.entry_price
                elif self.take_profit[1] == "Pip": # e.g. (2->$2*0.0001 +- EntryPrice, 'Pip', 0.0001)
                    price_delta = (self.take_profit[0]*self.take_profit[2])
                elif self.take_profit[1] == "Point": # e.g. (2->$2*0.01, "Point", $0.01)
                    price_delta = self.take_profit[0]*self.take_profit[2]
                else:
                    raise

                # Modify the Trade.sl property
                if trade.is_long:
                    trade.sl = trade.entry_price - price_delta
                if trade.is_short:
                    trade.sl = trade.entry_price + price_delta
                return

    def run_exit_when_pnl_lessthan(self):
        if self.exit_when_pnl_lessthan == None:
            return
        if self.exit_when_pnl_lessthan >= 0 or not isinstance(self.exit_when_pnl_lessthan, (float, int)):
            raise ValueError("ExitWhenPnLLessThan must be negative number.")
        if len(self.trades) == 0: # No open trades
            return
        if len(self.trades) > 1:
            raise Exception("The Backtester should only placed 1 trade at a time. There may be something wrong with your backtest settings.")
        trade = self.trades[0]

        if trade.pl <= self.exit_when_pnl_lessthan:
            trade.close()

    def update_tsl(self):
        if not self.enable_tsl: # Return method if TSL is not enabled
            return
        # We already asserted in init that the stop_loss is given.

        if len(self.trades) == 0: # No open trades
            return

        assert len(self.trades) == 1
        trade = self.trades[0]

        assert trade.sl != None, "The Stop-Loss of the Trade is not yet set."

        # Remind that we are at current bars close. When we make trades or updates, it will be executed on next bar's open
        if trade.is_long:
            # WARN: When using points, the value for new_sl can be zero or negative!
            new_sl = self.data.Close[-1-0] - self._price_delta # Current Close - Price Delta
            trade.sl = max(new_sl, trade.sl)
        if trade.is_short:
            new_sl = self.data.Close[-1-0] + self._price_delta # Current Close + Price Delta
            trade.sl = min(new_sl, trade.sl)

def evo_vbt_backtester(s_bool:SeriesBool, direction:str, df_ohlcv:pd.DataFrame,
                       exit_encoded_entry:bool = True,
                       vbt_params:dict=dict()
                       ) -> vbt.Portfolio:
    """The Original Backtester. This will be faster, but with limited features such as exits.

    Advantages:
    - Speed for Searching Optimal Signals
    - Faster for simulation in Stress-testing Stage

    Limitations:
    - Only time-exit I can implement is ExitAfterNBars.
    - Stop-Loss is fixed and as Percentage. Can't or I don't know how to implement Volatility Stop on entry.
    - It Can only exit based on exit_encoded_entry, exit_after_n_bars, fixed-SL, fixed-TP, fixed-TSL.
    - No Margin Trading (Margin=1. only).
    - Steep Learning Curve if you want more sophistication.

    Ideas:
    - Can be re-optimized using a different backtester like Backtest.py
        - Adding Extra Parameters and Optimizing (including different exits)
        - Walk-Forward Test & Optimization
    """


    _signal = s_bool.to_pd_series(index=df_ohlcv.index)

    if direction == 'LongOnly':
        _entries = _signal
        _exits = ~_signal
        _short_entries = None
        _short_exits = None
        _direction = Direction.LongOnly
        _upon_long_conflict = ConflictMode.Exit
        _upon_short_conflict = None
        _upon_dir_conflict = None
        _upon_opposite_entry = None

        _entries = _entries.shift(1).fillna(False)
        _exits = _exits.shift(1).fillna(False)
    if direction == 'ShortOnly':
        _entries = _signal
        _exits = ~_signal
        _short_entries = None
        _short_exits = None
        _direction = Direction.ShortOnly
        _upon_long_conflict = None
        _upon_short_conflict = ConflictMode.Exit
        _upon_dir_conflict = None
        _upon_opposite_entry = None

        _entries = _entries.shift(1).fillna(False)
        _exits = _exits.shift(1).fillna(False)
    if direction == 'LongShort':
        _entries = _signal
        _exits = None
        _short_entries = ~_signal
        _short_exits = None
        _direction = None
        _upon_long_conflict = None
        _upon_short_conflict = None
        _upon_dir_conflict = DirectionConflictMode.Opposite
        _upon_opposite_entry = OppositeEntryMode.Reverse

        _entries = _entries.vbt.fshift(1)#_entries.shift(1).fillna(False)
        _short_entries = _short_entries.vbt.fshift(1)#_short_entries.shift(1).fillna(False)

    # _entries = _entries.shift(1).fillna(False)
    # _exits = _exits.shift(1).fillna(False)
    # _short_entries = _short_entries.shift(1).fillna(False)
    # _short_exits = _short_exits.shift(1).fillna(False)

    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio'))
    vbt.settings.caching['whitelist'].extend([vbt.CacheCondition(base_cls='Portfolio', func='get_positions'), vbt.CacheCondition(base_cls='Portfolio', func='asset_flow')])

    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='cash_flow'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='asset_flow'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='get_filled_close'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='get_positions'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='assets'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='position_mask'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='position_coverage'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='get_init_cash'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='cash'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='gross_exposure'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='net_exposure'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='total_profit'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='final_value'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='total_return'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='get_returns_acc'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='benchmark_value'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='benchmark_returns'))
    # vbt.settings.caching['blacklist'].append(vbt.CacheCondition(base_cls='Portfolio', func='total_benchmark_return'))

    return vbt.Portfolio.from_signals(df_ohlcv['Close'], low=df_ohlcv['Low'], high=df_ohlcv['High'], open=None, price=df_ohlcv['Open'],
                                      entries=_entries,
                                      exits=_exits,
                                      short_entries=_short_entries,
                                      short_exits=_short_exits,
                                      direction=_direction,
                                      upon_long_conflict=_upon_long_conflict,
                                      upon_short_conflict=_upon_short_conflict,
                                      upon_dir_conflict=_upon_dir_conflict,
                                      upon_opposite_entry=_upon_opposite_entry,
                                      accumulate=False,
                                      **vbt_params
                                      )

def evo_backtester(ser_bool:SeriesBool, bt:Backtest, strat_params:dict=dict(),
                   splitter_func:callable=linear_is_oos,
                   cleaner_func:callable=backtesting_result_cleaner,
                   splitter_params:dict=dict(),
                   cleaner_params:dict=dict(),
                   use_pct_rets:bool=True
                   ) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
    """
    bt Backtest instance has to be instantiated outside the scope of this function, along with df_ohlcv and EvoStrategy.

    When bt is run with bt.run(..), it will return a pd.Series object with additional properties:
    - _trades : pd.DataFrame, containing the trades. This is to be cleaned by a cleaner function.
    - _equity_curve: pd.DataFrame, containing the Equity progression in dollar from the given initial cash and DrawdownPct.
        From this we can extract the percent or log returns from the dollar gains/losses. And use this as input for performance stat calculation (QuantStats or Empyrical)
    - _strategy : EvoStrategy(SignalStrategy), instance of EvoStrategy class.

    Notes:
    - We did not include the df_ohlcv, because we can access it in bt:Backtest instance.
    - Note that depending on the result, it's possible to return an empty Series or DataFrame. Check this in filtering and fitness calculations.

    Return
    Dictionary(str->Tuple[series_rets:pd.Series, df_trades:pd.DataFrame])

    Usage
    bt_res =  evo_backtester(...)
    bt_res["IS"](0) = pd.Series([....], index=<Depends on Splitter>)
    bt_res["OOS"](0) = pd.Series([....], index=<Depends on Splitter>)
    bt_res["ISOOS"](0) = pd.Series([....], index=<Depends on Splitter>)
    bt_res["IS"](1) = pd.DataFrame([....], index=<Depends on Splitter>)
    bt_res["OOS"](1) = pd.DataFrame([....], index=<Depends on Splitter>)
    bt_res["ISOOS"](1) = pd.DataFrame([....], index=<Depends on Splitter>)
    """

    strat_params.update(ser_bool=ser_bool) # Setting the value for ser_bool & direction

    import time
    # start_time = time.time()
    res = bt.run(**strat_params) # IS-OOS Result
    # end_time = time.time()
    # print("Bt Run Time:", end_time - start_time, "seconds")

    # Clean the ._trades DataFrame
    if cleaner_func != None:
        df_trades = cleaner_func(res._trades, **cleaner_params)
    else:
        df_trades = copy.deepcopy(res._trades)

    # Have to convert EntryTime and ExitTime as date because its considered as dtype=datetime64[ns]
    try:
        df_trades['EntryTime'] = df_trades['EntryTime'].dt.date
        df_trades['ExitTime'] = df_trades['ExitTime'].dt.date
    except AttributeError as err: # EntryTime and ExitTime may already be a date.
        pass

    # s_dd = res._equity_curve['DrawdownPct']  # Drawdown Series

    # Get the Returns (Pct Chg or Log)
    if use_pct_rets:
        s_rets = res._equity_curve['Equity'].pct_change().fillna(0.) # Pct Return Series
    else: # Use log returns:
        try:
            s_rets = res._equity_curve['Equity']
            s_rets = np.log(s_rets / s_rets.shift(1)).fillna(0.) # Log Return Series
        except ZeroDivisionError as err:
            # Use Percentage Return if catched ZeroDivisionError.
            s_rets = res._equity_curve['Equity'].pct_change().fillna(0.)  # Pct Return Series

    # Check if splitter_func is valid
    if splitter_func not in [linear_is_oos, multi_linear_is_oos]: raise ValueError("The Splitter Function has to be valid.")

    if splitter_func == linear_is_oos: # ((is_start_date, is_end_date), (oos_start_date, oos_end_date))
        out_dict = dict()
        tup_is, tup_oos = splitter_func(bt._data, **splitter_params)
        ind_is = bt._data.loc[tup_is[0]:tup_is[1]].index
        ind_oos = bt._data.loc[tup_oos[0]:tup_oos[1]].index
        # Use these indexes to get IS values from s_rets, s_dd, and df_trades. For df_trades, make sure to use set_index and reset_index
        out_dict["IS"] = (s_rets.loc[ind_is], df_trades[(df_trades['EntryTime']>=tup_is[0]) & (df_trades['EntryTime']<tup_is[1])])
        # out_dict["OOS"] = (s_rets.loc[ind_oos], df_trades[(df_trades['EntryTime'] >= tup_oos[0]) & (df_trades['EntryTime'] <= tup_oos[1])])
        out_dict["OOS"] = (s_rets.loc[ind_oos], df_trades[(df_trades['EntryTime'] >= tup_oos[0])])

    # Careful because IS's and OOS's, if filtered in pandas they are 'stitched' but performance stat calculation may throw errors!
    # The IS's and OOS's has to be appended.
    if splitter_func == multi_linear_is_oos: # out_dict[0]["IS"][0], out_dict[0]["IS"][1], out_dict[0]["OOS"][0], out_dict[0]["OOS"][1]
        out_dict = dict()
        out_dict["IS"] = [pd.Series(dtype=float), pd.DataFrame()] # Create empty Series & DataFrames, we will append items in the loop.
        out_dict["OOS"] = [pd.Series(dtype=float), pd.DataFrame()] # Create empty Series & DataFrames, we will append items in the loop.
        # To be converted back to tuple.
        ml_isoos_dict = splitter_func(bt._data, **splitter_params)

        for _, split_dict in ml_isoos_dict.items():
            for mode, tup in split_dict.items(): #e.g. ("IS", (start, end)) or ("OOS", (start, end))
                if mode == "IS":
                    out_dict[mode][0] = pd.concat([out_dict["IS"][0], s_rets.loc[tup[0]:tup[1]]], ignore_index=False) # Series Return
                    out_dict[mode][1] = pd.concat([out_dict["IS"][1], df_trades[(df_trades['EntryTime'] >= tup[0]) & (df_trades['EntryTime'] <= tup[1])]], ignore_index=False) # Trades DataFrame
                if mode == "OOS":
                    out_dict[mode][0] = pd.concat([out_dict["OOS"][0], s_rets.loc[tup[0]:tup[1]]], ignore_index=False) # Series Return
                    out_dict[mode][1] = pd.concat([out_dict["OOS"][1], df_trades[(df_trades['EntryTime'] >= tup[0]) & (df_trades['EntryTime'] <= tup[1])]], ignore_index=False) # Trades DataFrame
        out_dict["IS"] = tuple(out_dict["IS"]) # Convert to tuple
        out_dict["OOS"] = tuple(out_dict["OOS"]) # Convert to tuple
    out_dict["ISOOS"] = (s_rets, df_trades)

    # IS, OOS, ISOOS Results
    return out_dict

def evo_filter_layer1(ser_rets:pd.Series, df_trades:pd.DataFrame) -> bool:
    """This function can be applied to IS, OOS, ISOOS.

    Notes
    - This is optional for the user to be passed on gp_evaluator or evo_backtester.

    Filters:
    - Few Number of Trades (But what defines few?)
    - Anomaly in Returns
    - It only have one trade
    - The Strategy is bankrupt, meaning
    """

    try:
        # Sometimes pd.Series length could be 0, 1.
        outliers = quantstats.stats.outliers(ser_rets, quantile=.99) #pd.Series
    except ValueError as err:
        print(err)
    except Exception as err:
        print(err)
    else:
        # print("Number of Outliers:", outliers.shape[0])
        pass

    if (df_trades.shape[0] < 30): # Number of Trades must be at least 30 samples, otherwise return False
        return False
    if df_trades.empty or ser_rets.empty: # If the input Return Series or Trades DataFrame is empty.
        return False
    if (1 + ser_rets).cumprod()[-1] <= 0.05: # Lost Money or Bankrupt, False if Final Cumulative return is less than 100% of our equity, i.e We didnt make money at all
        return False
    if not outliers.empty and outliers[outliers > 0.].shape[0] > 100: # Nothing wrong with winning, but if there are extremely big winners it will affect some value of performance stats.
        return False

    # Returns True if All Filters are Passed
    return True

def evo_filter_layer2(bt_res:Dict[str, Tuple[pd.Series, pd.DataFrame]], filter_list) -> bool:
    """
    Input:
    bt_res : Dictionary(str->Tuple[series_rets:pd.Series, df_trades:pd.DataFrame])
        Backtest Result from evo_backtester e.g. bt_res["IS"][0] = Return Series & bt_res["IS"][1] = Trades DataFrame

    filter_list : List[Tuple[str, str, str, float, dict]],
        Tuple[IS-OOS mode, Perf Stat, Operator, Value, Perf Stat Parameters]

    Tuple in filter_list should have:
        mode:str, ['IS', 'OOS', 'ISOOS']
        perf_stat:str, ['Sharpe', 'Calmar', 'MaxDD', 'NumOfTrades', ...] (this will be mapped to a function because some may require parameters)
        operator:'str', ['gt', 'ge', 'lt', 'le', 'eq', 'ne']
        val:float|int, Value of Performance for comparison
        perf_stat_params:dict, Parameters for the chosen Performance Statistics, some may require such as benchmark for alpha & beta

    Usage:
    TODO: Update documentation evo_filter_layer2((‘OOS’, ‘Stability’, ‘gt’, 0.9, param_dict), (‘IS’, ‘Calmar’, ‘gt’, 0.1, {}), (‘ISOOS’, ‘Number of Trades’, ‘ge’, 500, param_dict))

    There are cases where performance stat require benchmark such as alpha, beta. In general, there are cases where performance stats are parametrized.
    """

    _OP = {"gt":operator.gt,
           "ge":operator.ge,
           "lt":operator.lt,
           "le":operator.le,
           "eq":operator.eq,
           "ne":operator.ne
           }
    _PS = {"Sharpe":quantstats.stats.sharpe,
           "Calmar":quantstats.stats.calmar,
           "Sortino":quantstats.stats.sortino,
           "CAGR/AvgDD": lambda rets: quantstats.stats.cagr(rets) / quantstats.stats.to_drawdown_series(rets).abs().mean(),
           "Stability":empyrical.stability_of_timeseries,
           # "AvgMonthlyReturns": lambda rets: quantstats.stats.monthly_returns(rets, eoy=False)['Month'].mean(),
           "Volatility":quantstats.stats.volatility,
           "VaR":quantstats.stats.value_at_risk,
           "CVaR":quantstats.stats.conditional_value_at_risk,
           "MaxDD":empyrical.max_drawdown,
           "AvgDD":lambda rets: quantstats.stats.to_drawdown_series(rets).mean(),
           "MaxDD_Duration":lambda rets: quantstats.stats.drawdown_details(quantstats.stats.to_drawdown_series(rets))['days'].max(),
           "Avg$PnL":lambda df: df['PnL'].mean(),
           "Avg$Loss": lambda df: df['PnL'].where(df['PnL'] < 0.).mean(),
           "Avg$Profit": lambda df: df['PnL'].where(df['PnL'] > 0.).mean(),
           "NumberOfTrades":lambda df: df.shape[0],
           "Total$PnL": lambda df: df['PnL'].sum(),
           "Max$Loss":lambda df: df['PnL'].min()
           }
    # Performance Stat Function Requires: 0=Returns Series, 1=Trade DataFrame
    _PS_IN = \
    {"Sharpe": 0,
     "Calmar": 0,
     "Sortino": 0,
     "CAGR/AvgDD": 0,
     "Stability": 0,
     # "AvgMonthlyReturns": 0,
     "Volatility":0,
     "VaR":0,
     "CVaR":0,
     "MaxDD": 0,
     "AvgDD": 0,
     "MaxDD_Duration":0,
     "Avg$PnL": 1,
     "Avg$Loss": 1,
     "Avg$Profit": 1,
     "NumberOfTrades":1,
     "Total$PnL":1,
     "Max$Loss":1
     }

    # In some cases, the perf stats for IS is valid but not of OOS maybe because the window size for OOS is too small.
    for tup in filter_list: # (IS-OOS mode, Perf Stat, Operator, Value, Perf Stat Parameters)
        assert (tup[0] in ["IS","OOS","ISOOS"]) and (tup[1] in _PS.keys()) and (tup[2] in _OP) and (isinstance(tup[3], (float,int))) and (isinstance(tup[4], dict))
        try:
            if _PS_IN[tup[1]] == 0:
                f = _OP[tup[2]](_PS[tup[1]](bt_res[tup[0]][0], **tup[4]), tup[3])
                if not f: return False
            if _PS_IN[tup[1]] == 1:
                f = _OP[tup[2]](_PS[tup[1]](bt_res[tup[0]][1], **tup[4]), tup[3])
                if not f: return False
        except ZeroDivisionError: # If it catches any kind of errors including ZeroDivisionError and OverflowError
            print("We catched a ZeroDivisionError. We will return False in 2nd Filter Layer.")
            return False
        except OverflowError:
            print("We catched an OverflowError. We will return False in 2nd Filter Layer.")
            return False
        except Exception as err:
            print(type(err))
            print(err)
            return False
    return True

# TODO: Implement 3rd Filter Layer.
def evo_filter_layer3():
    return # Returns True if Passed, else False

__all__ = ["EvoStrategy", "evo_backtester", "evo_filter_layer1", "evo_filter_layer2", "evo_filter_layer3"]