from evoquant.base import *
from evoquant.signals import *

import pandas_ta as ta
import pandas as pd
import numpy as np

def shift(ser:SeriesFloat, lag:Lag) -> SeriesFloat:
    assert type(lag) == Lag
    res = SeriesFloat(ser.to_pd_series().shift(lag.value).reset_index(drop=True), name="Shift", na_value=np.nan)
    return res

def sum_indicator(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesFloat:
    res = SeriesFloat(ser1.to_pd_series()+ser2.to_pd_series(), name="ADD", na_value=np.nan)
    return res

def diff(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesFloat:
    return SeriesFloat(ser1.to_pd_series() - ser2.to_pd_series(), name="DIFF")

def abs_value(ser:SeriesFloat) -> SeriesFloat:
    return SeriesFloat(ser.to_pd_series().abs(), name="ABS")

def abs_diff(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesFloat:
    return SeriesFloat((ser1.to_pd_series() - ser2.to_pd_series()).abs(), name="AbsDiff")

def sma(ser:SeriesFloat, period:Period) -> SeriesFloat:
    """Simple Moving Average Indicator
    """
    assert type(period) == Period
    # return ser.to_pd_series().rolling(period.value).mean()
    return SeriesFloat(ta.sma(ser.to_pd_series(), length=period.value), name="SMA")

def highest(ser:SeriesFloat, period:Period) -> SeriesFloat:
    return SeriesFloat(ser.to_pd_series().rolling(period.value).max(), name="Highest")

def lowest(ser:SeriesFloat, period:Period) -> SeriesFloat:
    return SeriesFloat(ser.to_pd_series().rolling(period.value).min(), name="Lowest")
########################################################################################################################
# Idea: We can make rsi function as a Static Method in RSI e.g. RSI.main_func(ser:SeriesFloat, period:Period)

class RSI(SeriesFloat, SeriesBase):
    # Need to implement this because rsi is a stationary series.
    def __init__(self, in_series, name="RSI", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

def rsi(ser:SeriesFloat, period:Period) -> SeriesFloat:
    return SeriesFloat(ta.rsi(ser.to_pd_series(), length=period.value, scalar=False), name="RSI")

class RSIValue(ParameterBase):
    def __init__(self, in_value, param_name="RSIValue", param_type="Output", str_style=5):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, float):
            if (not isinstance(in_value, int)) or isinstance(in_value, bool):
                raise TypeError("Value must be an float or integer.")
            else:
                self._value = float(in_value)
        if in_value < 0. or in_value > 100.:
            raise ValueError(f"RSIValue must be within 0 & 100")

COMPILE_RSI = \
r"""
for comb in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(rsi, [comb, Period], SeriesFloat, name="rsi")
    pset.addPrimitive(rsi, [comb, Period], SeriesIndicator, name="rsi")
    pset.addPrimitive(rsi, [comb, Period], RSI, name="rsi")
pset.addPrimitive(rsi, [SeriesVolume, Period], SeriesIndicator, name="rsi")

pset.addEphemeralConstant("RSIValue", lambda: RSIValue(round(random.uniform(0.01, 99.99), 2)), RSIValue)

pset.addPrimitive(series_above_quantile_rule, [RSI, Quantile, Period], SeriesBool, name="series_above_quantile_rule")
pset.addPrimitive(series_below_quantile_rule, [RSI, Quantile, Period], SeriesBool, name="series_below_quantile_rule")
pset.addPrimitive(series_cross_above_quantile_rule, [RSI, Quantile, Period], SeriesBool, name="series_cross_above_quantile_rule")
pset.addPrimitive(series_cross_below_quantile_rule, [RSI, Quantile, Period], SeriesBool, name="series_cross_below_quantile_rule")

pset.addPrimitive(series_above_ma_rule, [RSI, Period], SeriesBool, name="series_above_ma_rule")
pset.addPrimitive(series_below_ma_rule, [RSI, Period], SeriesBool, name="series_below_ma_rule")
pset.addPrimitive(series_cross_above_ma_rule, [RSI, Period], SeriesBool, name="series_cross_above_ma_rule")
pset.addPrimitive(series_cross_below_ma_rule, [RSI, Period], SeriesBool, name="series_cross_below_ma_rule")

pset.addPrimitive(series_above_shift_rule, [RSI, Lag], SeriesBool, name="series_above_shift_rule")
pset.addPrimitive(series_below_shift_rule, [RSI, Lag], SeriesBool, name="series_below_shift_rule")
pset.addPrimitive(series_cross_above_shift_rule, [RSI, Lag], SeriesBool, name="series_cross_above_shift_rule")
pset.addPrimitive(series_cross_below_shift_rule, [RSI, Lag], SeriesBool, name="series_cross_below_shift_rule")

pset.addPrimitive(series_cross_above_value_rule, [RSI, RSIValue], SeriesBool, name="series_cross_above_value_rule")
pset.addPrimitive(series_cross_below_value_rule, [RSI, RSIValue], SeriesBool, name="series_cross_below_value_rule")
pset.addPrimitive(series_above_value_rule, [RSI, RSIValue], SeriesBool, name="series_above_value_rule")
pset.addPrimitive(series_below_value_rule, [RSI, RSIValue], SeriesBool, name="series_below_value_rule")

if Quantile not in terminal_types: terminal_types.append(Quantile) 
if Period not in terminal_types: terminal_types.append(Period)
if Lag not in terminal_types: terminal_types.append(Lag)
if RSIValue not in terminal_types: terminal_types.append(RSIValue)
"""
#-----------------------------------------------------------------------------------------------------------------------
class ZScore(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="ZScore", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

def zscore(ser:SeriesFloat, period:Period) -> SeriesFloat:
    rolling_mean = ser.to_pd_series().rolling(window=period.value).mean()
    rolling_std = ser.to_pd_series().rolling(window=period.value).std()
    # Calculate the z-score
    _zscore = (ser.to_pd_series() - rolling_mean) / rolling_std
    return SeriesFloat(_zscore, name="ZScore", na_value=np.nan)

class ZScoreValue(ParameterBase):
    def __init__(self, in_value, param_name="ZScoreValue", param_type="Output", str_style=5):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, float):
            if (not isinstance(in_value, int)) or isinstance(in_value, bool):
                raise TypeError("Value must be an float or integer.")
            else:
                self._value = float(in_value)
        # if in_value < -6. or in_value > 6.:
        #     raise ValueError(f"ZScoreValue must be within -6 & 6")

COMPILE_ZSCORE = \
r"""
for comb in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(zscore, [comb, Period], SeriesFloat, name="zscore")
    pset.addPrimitive(zscore, [comb, Period], SeriesIndicator, name="zscore")
    pset.addPrimitive(zscore, [comb, Period], ZScore, name="zscore")
pset.addPrimitive(zscore, [SeriesVolume, Period], ZScore, name="zscore")

pset.addEphemeralConstant("ZScoreValue", lambda: ZScoreValue(round(random.uniform(-3., 3.), 2)), ZScoreValue)

pset.addPrimitive(series_above_quantile_rule, [ZScore, Quantile, Period], SeriesBool, name="series_above_quantile_rule")
pset.addPrimitive(series_below_quantile_rule, [ZScore, Quantile, Period], SeriesBool, name="series_below_quantile_rule")
pset.addPrimitive(series_cross_above_quantile_rule, [ZScore, Quantile, Period], SeriesBool, name="series_cross_above_quantile_rule")
pset.addPrimitive(series_cross_below_quantile_rule, [ZScore, Quantile, Period], SeriesBool, name="series_cross_below_quantile_rule")

pset.addPrimitive(series_above_ma_rule, [ZScore, Period], SeriesBool, name="series_above_ma_rule")
pset.addPrimitive(series_below_ma_rule, [ZScore, Period], SeriesBool, name="series_below_ma_rule")
pset.addPrimitive(series_cross_above_ma_rule, [ZScore, Period], SeriesBool, name="series_cross_above_ma_rule")
pset.addPrimitive(series_cross_below_ma_rule, [ZScore, Period], SeriesBool, name="series_cross_below_ma_rule")

pset.addPrimitive(series_above_shift_rule, [ZScore, Lag], SeriesBool, name="series_above_shift_rule")
pset.addPrimitive(series_below_shift_rule, [ZScore, Lag], SeriesBool, name="series_below_shift_rule")
pset.addPrimitive(series_cross_above_shift_rule, [ZScore, Lag], SeriesBool, name="series_cross_above_shift_rule")
pset.addPrimitive(series_cross_below_shift_rule, [ZScore, Lag], SeriesBool, name="series_cross_below_shift_rule")

pset.addPrimitive(series_cross_above_value_rule, [ZScore, ZScoreValue], SeriesBool, name="series_cross_above_value_rule")
pset.addPrimitive(series_cross_below_value_rule, [ZScore, ZScoreValue], SeriesBool, name="series_cross_below_value_rule")
pset.addPrimitive(series_above_value_rule, [ZScore, ZScoreValue], SeriesBool, name="series_above_value_rule")
pset.addPrimitive(series_below_value_rule, [ZScore, ZScoreValue], SeriesBool, name="series_below_value_rule")

pset.addPrimitive(cross_above_rule, [ZScore, ZScore], SeriesBool, name="cross_above_rule")
pset.addPrimitive(cross_below_rule, [ZScore, ZScore], SeriesBool, name="cross_below_rule")
pset.addPrimitive(is_above_rule, [ZScore, ZScore], SeriesBool, name="is_above_rule")
pset.addPrimitive(is_below_rule, [ZScore, ZScore], SeriesBool, name="is_below_rule")

if Quantile not in terminal_types: terminal_types.append(Quantile) 
if Period not in terminal_types: terminal_types.append(Period)
if Lag not in terminal_types: terminal_types.append(Lag)
if ZScoreValue not in terminal_types: terminal_types.append(ZScoreValue)
"""

#-----------------------------------------------------------------------------------------------------------------------
class StdDev(ParameterBase):
    def __init__(self, in_value, param_name="std", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, float):
            if (not isinstance(in_value, int)) or isinstance(in_value, bool):
                raise TypeError("Value must be an float or integer.")
            else:
                self._value = float(in_value)
        if in_value <= 0.:
            raise ValueError(f"Std must be more than zero.")
class MAMode(ParameterBase):
    """
    [MAMode('ema'), MAMode('sma')]
    """
    # The Value in the pair must be a Raw Syntax in cTrader C#
    _CTRADER = \
        {'ema': r'MovingAverageType.Exponential',
         'sma': r'MovingAverageType.Simple'
         }
    _MT5 = {}
    def __init__(self, in_value, param_name="mamode", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, str):
            raise TypeError("Value must be a string.")
        if in_value not in ['ema', 'sma']:
            raise ValueError(r"MAMode must be either 'ema' or 'sma'")
    def value_mapper(self, trade_platform): # Override value_mapper method
        if trade_platform == "ctrader":
            return self._CTRADER[self.value]

class BBOut(ParameterBase):
    """
    [BBOut('bbl'), BBOut('bbm'), BBOut('bbu'), BBOut('bbb'), BBOut('bbp')]
    """
    # The Value in the pair must be a Raw Syntax in cTrader C#
    _CTRADER = \
        {
            'bbl':r'BBL',
            'bbm':r'BBM',
            'bbu': r'BBU',
            'bbb': r'BBB',
            'bbp': r'BBP',
        }
    def __init__(self, in_value, param_name="bbout", param_type="Output", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, str):
            raise TypeError("Value must be a string.")
        if in_value not in ['bbl', 'bbm', 'bbu', 'bbb', 'bbp']:
            raise ValueError(r"BBOut must be either: 'bbl', 'bbm', 'bbu', 'bbb', 'bbp'")
    def value_mapper(self, trade_platform):
        if trade_platform == "ctrader":
            return self._CTRADER[self.value]

def bbands(ser:SeriesFloat, period:Period, std:StdDev, mamode:MAMode, bbout:BBOut) -> SeriesFloat:
    if bbout.value == 'bbl':
        return SeriesFloat(ta.bbands(ser.to_pd_series(), length=period.value, std=std.value, mamode=mamode.value).iloc[:,0], name="BBandsBBL")
    if bbout.value == 'bbm':
        return SeriesFloat(ta.bbands(ser.to_pd_series(), length=period.value, std=std.value, mamode=mamode.value).iloc[:,1], name="BBandsBBM")
    if bbout.value == 'bbu':
        return SeriesFloat(ta.bbands(ser.to_pd_series(), length=period.value, std=std.value, mamode=mamode.value).iloc[:,2], name="BBandsBBU")
    if bbout.value == 'bbb':
        return SeriesFloat(ta.bbands(ser.to_pd_series(), length=period.value, std=std.value, mamode=mamode.value).iloc[:,3], name="BBandsBBB")
    if bbout.value == 'bbp':
        return SeriesFloat(ta.bbands(ser.to_pd_series(), length=period.value, std=std.value, mamode=mamode.value).iloc[:,4], name="BBandsBBP")
#-----------------------------------------------------------------------------------------------------------------------

########################################################################################################################
import inspect
def list_module_contents():
    module_members = globals().items()
    module_name = globals()['__name__']

    classes = []
    functions = []
    variables = []

    for name, obj in module_members:
        if inspect.isclass(obj) and obj.__module__ == module_name:
            classes.append(name)
        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append(name)
        elif not inspect.ismodule(obj) and not inspect.isclass(obj) and not inspect.isfunction(obj):
            variables.append(name)
    return classes+[x for x in functions if x != 'list_module_contents']
__all__  = list_module_contents()

# for i in __all__:
#     print(i)

# d = pd.read_csv("D:\Projects\FinDashAnalytics\Data\ASX OHLCV\HVN.csv")
# s = SeriesFloat(d["Close"])

# print(bbands(s, Period(5), StdDev(2), MAMode('sma'), bbout=BBOut('bbp')))
# print(s.to_pd_series())
# print(shift(s, Lag(2)).series)