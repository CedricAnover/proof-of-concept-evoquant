from evoquant.base import *

import pandas as pd
import numpy as np

def _shift_array(arr, window_size, direction='right'):
    if window_size > len(arr):
        raise ValueError("Window size cannot be larger than the array size.")

    if direction == 'right':
        shifted_arr = np.concatenate((np.full(window_size, np.nan), arr[:-window_size]))
    elif direction == 'left':
        shifted_arr = np.concatenate((arr[window_size:], np.full(window_size, np.nan)))
    else:
        raise ValueError("Invalid direction. Choose either 'right' or 'left'.")

    return shifted_arr

def and_rule(ser1:SeriesBool, ser2:SeriesBool)-> SeriesBool:
    # res = ser1.to_pd_series().reset_index(drop=True) & ser2.to_pd_series().reset_index(drop=True)
    res = np.logical_and(ser1.series, ser2.series)
    return SeriesBool(res, name="AND", na_value=False)

def and_rule3(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) & \
          ser2.to_pd_series().reset_index(drop=True) & \
          ser3.to_pd_series().reset_index(drop=True)
    # res = np.logical_and(ser1.series, ser2.series, ser3.series)
    return SeriesBool(res, name="AND3", na_value=False)

def and_rule4(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool, ser4:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) & \
          ser2.to_pd_series().reset_index(drop=True) & \
          ser3.to_pd_series().reset_index(drop=True) & \
          ser4.to_pd_series().reset_index(drop=True)
    # res = np.logical_and(np.logical_and(ser1.series, ser2.series), np.logical_and(ser3.series, ser4.series))
    return SeriesBool(res, name="AND4", na_value=False)

def or_rule(ser1:SeriesBool, ser2:SeriesBool)-> SeriesBool:
    res = np.logical_or(ser1.series, ser2.series)
    # res = ser1.to_pd_series().reset_index(drop=True) | ser2.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="OR", na_value=False)

def or_rule3(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) | \
          ser2.to_pd_series().reset_index(drop=True) | \
          ser3.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="OR3", na_value=False)

def or_rule4(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool, ser4:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) | \
          ser2.to_pd_series().reset_index(drop=True) | \
          ser3.to_pd_series().reset_index(drop=True) | \
          ser4.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="OR4", na_value=False)

def xor_rule(ser1:SeriesBool, ser2:SeriesBool)-> SeriesBool:
    # assert ser1.size == ser2.size
    res = np.logical_xor(ser1.series, ser2.series)
    # res = ser1.to_pd_series().reset_index(drop=True) ^ ser2.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="XOR", na_value=False)

def xor_rule3(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) ^ \
          ser2.to_pd_series().reset_index(drop=True) ^ \
          ser3.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="XOR3", na_value=False)

def xor_rule4(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool, ser4:SeriesBool)-> SeriesBool:
    res = ser1.to_pd_series().reset_index(drop=True) ^ \
          ser2.to_pd_series().reset_index(drop=True) ^ \
          ser3.to_pd_series().reset_index(drop=True) ^ \
          ser4.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="XOR4", na_value=False)

def not_rule(ser:SeriesBool) -> SeriesBool:
    res = np.logical_not(ser.series)
    # res = ~ser.to_pd_series().reset_index(drop=True)
    return SeriesBool(res, name="NOT", na_value=False)
#-----------------------------------------------------------------------------------------------------------------------
def and2_or1(ser1:SeriesBool, ser2:SeriesBool, ser3:SeriesBool, ser4:SeriesBool) -> SeriesBool:
    and1 = and_rule(ser1, ser2)
    and2 = and_rule(ser3, ser4)
    return or_rule(and1, and2)

def and3_or1(*args) -> SeriesBool:
    assert len(args) == 6, "There must be 6 arguments for and3_or1 signal primitive"
    assert all([type(elem)==SeriesBool for elem in  args]), "All arguments of and3_or1 must be SeriesBool"
    return or_rule(and_rule3(*args[:3]), and_rule3(*args[3:]))

def and2_or2(*args) -> SeriesBool:
    assert len(args) == 6, "There must be 6 arguments for and2_or2 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of and2_or2 must be SeriesBool"
    return or_rule3(and_rule(*args[:2]), and_rule(*args[2:4]), and_rule(*args[4:]))

def and3_or2(*args) -> SeriesBool:
    assert len(args) == 9, "There must be 9 arguments for and3_or2 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of and3_or2 must be SeriesBool"
    return or_rule3(and_rule3(*args[:3]), and_rule3(*args[3:6]), and_rule3(*args[6:]))

def and2_or3(*args) -> SeriesBool:
    assert len(args) == 8, "There must be 8 arguments for and2_or3 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of and2_or3 must be SeriesBool"
    return or_rule4(and_rule(*args[:2]), and_rule(*args[2:4]), and_rule(*args[4:6]), and_rule(*args[6:]))

def and3_or3(*args) -> SeriesBool:
    assert len(args) == 12, "There must be 12 arguments for and3_or3 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of and3_or3 must be SeriesBool"
    return or_rule4(and_rule3(*args[:3]), and_rule3(*args[3:6]), and_rule3(*args[6:9]), and_rule3(*args[9:]))

#-----------------------------------------------------------------------------------------------------------------------
def or2_and1(*args) -> SeriesBool:
    assert len(args) == 4, "There must be 4 arguments for or2_and1 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or2_and1 must be SeriesBool"
    return and_rule(or_rule(*args[:2]), or_rule(*args[2:]))

def or3_and1(*args) -> SeriesBool:
    assert len(args) == 6, "There must be 6 arguments for or3_and1 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or3_and1 must be SeriesBool"
    return and_rule(or_rule3(*args[:3]), or_rule3(*args[3:]))

def or4_and1(*args) -> SeriesBool:
    assert len(args) == 8, "There must be 8 arguments for or4_and1 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or4_and1 must be SeriesBool"
    return and_rule(or_rule4(*args[:4]), or_rule4(*args[4:]))

def or2_and2(*args) -> SeriesBool:
    assert len(args) == 6, "There must be 6 arguments for or2_and2 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or2_and2 must be SeriesBool"
    return and_rule3(or_rule(*args[:2]), or_rule(*args[2:4]), or_rule(*args[4:6]))

def or3_and2(*args) -> SeriesBool:
    assert len(args) == 9, "There must be 9 arguments for or3_and2 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or3_and2 must be SeriesBool"
    return and_rule3(or_rule3(*args[:3]), or_rule3(*args[3:6]), or_rule3(*args[6:]))

def or4_and2(*args) -> SeriesBool:
    assert len(args) == 12, "There must be 12 arguments for or4_and2 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or4_and2 must be SeriesBool"
    return and_rule3(or_rule4(*args[:4]), or_rule4(*args[4:8]), or_rule4(*args[8:]))

def or2_and3(*args) -> SeriesBool:
    assert len(args) == 8, "There must be 8 arguments for or2_and3 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or2_and3 must be SeriesBool"
    return and_rule4(or_rule(*args[:2]), or_rule(*args[2:4]), or_rule(*args[4:6]), or_rule(*args[6:]))

def or3_and3(*args) -> SeriesBool:
    assert len(args) == 12, "There must be 12 arguments for or3_and3 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or3_and3 must be SeriesBool"
    return and_rule4(or_rule3(*args[:3]), or_rule3(*args[3:6]), or_rule3(*args[6:9]), or_rule3(*args[9:]))

def or4_and3(*args) -> SeriesBool:
    assert len(args) == 16, "There must be 16 arguments for or4_and3 signal primitive"
    assert all([type(elem) == SeriesBool for elem in args]), "All arguments of or4_and3 must be SeriesBool"
    return and_rule4(or_rule4(*args[:4]), or_rule4(*args[4:8]), or_rule4(*args[8:12]), or_rule4(*args[12:]))
#-----------------------------------------------------------------------------------------------------------------------
def cross_above_rule(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesBool:
    # current = ser1.to_pd_series().reset_index(drop=True) > ser2.to_pd_series().reset_index(drop=True)
    # previous = ser1.to_pd_series().reset_index(drop=True).shift(1) < ser2.to_pd_series().reset_index(drop=True).shift(1)

    current = ser1.series > ser2.series
    previous = _shift_array(ser1.series, 1) < _shift_array(ser2.series, 1)

    cross = np.logical_and(current, previous)

    # cross = current & previous
    return SeriesBool(cross, name="CrossAbove", na_value=False)
def cross_below_rule(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesBool:
    # current = ser1.to_pd_series().reset_index(drop=True) < ser2.to_pd_series().reset_index(drop=True)
    # previous = ser1.to_pd_series().reset_index(drop=True).shift(1) > ser2.to_pd_series().reset_index(drop=True).shift(1)

    current = ser1.series < ser2.series
    previous = _shift_array(ser1.series, 1) > _shift_array(ser2.series, 1)

    cross = np.logical_and(current, previous)
    # cross = current & previous
    return SeriesBool(cross, name="CrossBelow", na_value=False)

def is_above_rule(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesBool:
    # res = SeriesBool(ta.above(ser1.to_pd_series(), ser2.to_pd_series(), asint=False), name="IsAbove", na_value=False)
    # res = ser1.to_pd_series().reset_index(drop=True) > ser2.to_pd_series().reset_index(drop=True)
    res = ser1.series > ser2.series
    return SeriesBool(res, name="IsAbove", na_value=False)

def is_below_rule(ser1:SeriesFloat, ser2:SeriesFloat) -> SeriesBool:
    # res = SeriesBool(ta.below(ser1.to_pd_series(), ser2.to_pd_series(), asint=False), name="IsBelow", na_value=False)
    # res = ser1.to_pd_series().reset_index(drop=True) < ser2.to_pd_series().reset_index(drop=True)
    res = ser1.series < ser2.series
    return SeriesBool(res, name="IsBelow", na_value=False)
#-----------------------------------------------------------------------------------------------------------------------
"""
For rules that compares a Series to a constant value, in deap gp, we have to use Ephemiral Constants that generates
random from [0,100] [0,1] [-100,100] [-1,1]
"""
# def cross_above_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     s = ser.to_pd_series()
#     # v_s = pd.Series([val] * len(s))
#     len_s = len(s)
#     return SeriesBool(ta.cross(s, pd.Series([val] * len_s), above=True, asint=False), name="CrossAboveValue", na_value=False)
# def cross_below_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     s = ser.to_pd_series()
#     # v_s = pd.Series([val] * len(s))
#     len_s = len(s)
#     return SeriesBool(ta.cross(s, pd.Series([val] * len_s), above=False, asint=False), name="CrossBelowValue", na_value=False)

# def cross_above_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     val_s = pd.Series(val, index=range(ser.size), dtype=float, name=str(val))
#     cross = cross_above_rule(ser, SeriesFloat(val_s)) #->SeriesBool
#     return cross
# def cross_below_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     val_s = pd.Series(val, index=range(ser.size), dtype=float, name=str(val))
#     cross = cross_below_rule(ser, SeriesFloat(val_s))  # ->SeriesBool
#     return cross
# def is_above_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     return SeriesBool(ser.series > val, name="IsAboveValue", na_value=False)
# def is_below_value_rule(ser:SeriesFloat, val:float) -> SeriesBool:
#     return SeriesBool(ser.series < val, name="IsAboveValue", na_value=False)
#-----------------------------------------------------------------------------------------------------------------------
class IncrDecrNBars(ParameterBase):
    """
    [IncrDecrNBars(min_n_bars),...,IncrDecrNBars(max_n_bars)]
    IncrDecrNBars(2),IncrDecrNBars(3),...,IncrDecrNBars(5)
    """
    def __init__(self, in_value, param_name="incrdecr_n_bars", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be a integer.")
        if in_value < 2:
            raise ValueError(f"IncrDecrNBars must be at least 2.")
def is_incr_n_bars_rule(ser:SeriesFloat, nbars:IncrDecrNBars) -> SeriesBool:
    assert type(nbars) == IncrDecrNBars
    return SeriesBool(ser.to_pd_series().rolling(window=nbars.value).apply(lambda x: all(x[i] > x[i - 1] for i in range(1, nbars.value)), raw=True), name="IsIncr", na_value=False)
def is_decr_n_bars_rule(ser:SeriesFloat, nbars:IncrDecrNBars) -> SeriesBool:
    assert type(nbars) == IncrDecrNBars
    return SeriesBool(ser.to_pd_series().rolling(window=nbars.value).apply(lambda x: all(x[i] < x[i - 1] for i in range(1, nbars.value)), raw=True), name="IsDecr", na_value=False)
#-----------------------------------------------------------------------------------------------------------------------
class HighLowNBars(ParameterBase):
    def __init__(self, in_value, param_name="highlow_n_bars", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be a integer.")
        if in_value < 2:
            raise ValueError(f"HighLowNBars must be at least 2.")
def is_highest_n_bars_rule(ser:SeriesFloat, nbars:HighLowNBars) -> SeriesBool:
    assert type(nbars) == HighLowNBars
    return SeriesBool(ser.to_pd_series().rolling(window=nbars.value).apply(lambda x: x[-1] == max(x), raw=True), name="IsHighest", na_value=False)
def is_lowest_n_bars_rule(ser:SeriesFloat, nbars:HighLowNBars) -> SeriesBool:
    assert type(nbars) == HighLowNBars
    return SeriesBool(ser.to_pd_series().rolling(window=nbars.value).apply(lambda x: x[-1] == min(x), raw=True), name="IsLowest", na_value=False)

# def is_series_higher_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
#     return SeriesBool(ser.to_pd_series().diff(periods=lag.value) > 0., name="SeriesHigherShift", na_value=False)
# def is_series_lower_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
#     return SeriesBool(ser.to_pd_series().diff(periods=lag.value) < 0., name="SeriesHigherShift", na_value=False)

#-----------------------------------------------------------------------------------------------------------------------
class DayOfWeek(ParameterBase):
    """
    1=Mon,...,7=Sun
    """
    _CTRADER = \
        {1: r"DayOfWeek.Monday",
         2: r"DayOfWeek.Tuesday",
         3: r"DayOfWeek.Wednesday",
         4: r"DayOfWeek.Thursday",
         5: r"DayOfWeek.Friday",
         6: r"DayOfWeek.Saturday",
         7: r"DayOfWeek.Sunday"}
    def __init__(self, in_value, param_name="DayOfWeekValue", param_type="Output", str_style=5, min_val=1, max_val=7):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be a integer.")
        if in_value < min_val or in_value > max_val:
            raise ValueError(f"DayOfWeek must be within {min_val} & {max_val}")

    def value_mapper(self, trade_platform):
        if trade_platform=="ctrader":
            return self._CTRADER[self.value]

class MonthInYear(ParameterBase):
    """
    1=Jan,...,12=Dec
    """
    def __init__(self, in_value, param_name="MonthInYearValue", param_type="Output", str_style=5, min_val=1, max_val=12):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be a integer.")
        if in_value < min_val or in_value > max_val:
            raise ValueError(f"MonthInYear must be within {min_val} & {max_val}")

class HourInDay(ParameterBase):
    """
    0 = 12am
    1 = 1am
    ...
    12 = 12pm
    ...
    23 = 11pm
    """
    def __init__(self, in_value, param_name="MonthInYearValue", param_type="Output", str_style=5, min_val=0, max_val=23):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be a integer.")
        if in_value < min_val or in_value > max_val:
            raise ValueError(f"HourInDay must be within {min_val} & {max_val}")

def day_of_week_rule(ser:SeriesDate, dow:DayOfWeek) -> SeriesBool:
    assert type(dow) == DayOfWeek
    return SeriesBool(np.equal(ser.day_of_week, dow.value), name="IsDayOfWeek", na_value=False)
def month_in_year_rule(ser:SeriesDate, miy:MonthInYear) -> SeriesBool:
    assert type(miy) == MonthInYear
    return SeriesBool(np.equal(ser.month_in_year, miy.value), name="IsMonthInYear", na_value=False)

def hour_in_day_rule(ser:SeriesDate, hid:HourInDay) -> SeriesBool:
    assert type(hid) == HourInDay
    return SeriesBool(np.equal(ser.hour_in_day, hid.value), name="IsHourInDay", na_value=False)
def hour_in_day_ge_rule(ser:SeriesDate, hid:HourInDay) -> SeriesBool:
    assert type(hid) == HourInDay
    return SeriesBool(np.greater_equal(ser.hour_in_day, hid.value), name="IsHourInDayGE", na_value=False)
def hour_in_day_le_rule(ser:SeriesDate, hid:HourInDay) -> SeriesBool:
    assert type(hid) == HourInDay
    return SeriesBool(np.less_equal(ser.hour_in_day, hid.value), name="IsHourInDayLE", na_value=False)
#-----------------------------------------------------------------------------------------------------------------------
class Quantile(ParameterBase):
    def __init__(self, in_value, param_name="quantile", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, float):
            if (not isinstance(in_value, int)) or isinstance(in_value, bool):
                raise TypeError("Value must be an float or integer.")
            else:
                self._value = float(in_value)
        if in_value <= 0. or in_value >= 1.:
            raise ValueError(f"Quantile must be within the open interval (0, 1).")
def series_above_quantile_rule(ser:SeriesFloat, q:Quantile, period:Period) -> SeriesBool:
    return SeriesBool(ser.to_pd_series() > ser.to_pd_series().rolling(period.value).quantile(q.value), name="AboveQuantile", na_value=False)
def series_below_quantile_rule(ser:SeriesFloat, q:Quantile, period:Period) -> SeriesBool:
    return SeriesBool(ser.to_pd_series() < ser.to_pd_series().rolling(period.value).quantile(q.value), name="BelowQuantile", na_value=False)
def series_cross_above_quantile_rule(ser:SeriesFloat, q:Quantile, period:Period) -> SeriesBool:
    s_q = ser.to_pd_series().rolling(period.value).quantile(q.value)
    res = cross_above_rule(ser, SeriesFloat(s_q)) # -> SeriesBool
    return res
def series_cross_below_quantile_rule(ser:SeriesFloat, q:Quantile, period:Period) -> SeriesBool:
    s_q = ser.to_pd_series().rolling(period.value).quantile(q.value)
    res = cross_below_rule(ser, SeriesFloat(s_q))  # -> SeriesBool
    return res
#--------
def series_above_ma_rule(ser:SeriesFloat, period:Period) -> SeriesBool:
    s_ma = ser.to_pd_series().rolling(period.value).mean()
    res = is_above_rule(ser, SeriesFloat(s_ma))
    return res
def series_below_ma_rule(ser:SeriesFloat, period:Period) -> SeriesBool:
    s_ma = ser.to_pd_series().rolling(period.value).mean()
    res = is_below_rule(ser, SeriesFloat(s_ma))
    return res
def series_cross_above_ma_rule(ser:SeriesFloat, period:Period) -> SeriesBool:
    s_ma = ser.to_pd_series().rolling(period.value).mean()
    res = cross_above_rule(ser, SeriesFloat(s_ma))
    return res
def series_cross_below_ma_rule(ser:SeriesFloat, period:Period) -> SeriesBool:
    s_ma = ser.to_pd_series().rolling(period.value).mean()
    res = cross_below_rule(ser, SeriesFloat(s_ma))
    return res
#--------
def series_above_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
    s_lag = ser.to_pd_series().shift(lag.value)
    res = is_above_rule(ser, SeriesFloat(s_lag))
    return res
def series_below_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
    s_lag = ser.to_pd_series().shift(lag.value)
    res = is_below_rule(ser, SeriesFloat(s_lag))
    return res
def series_cross_above_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
    s_lag = ser.to_pd_series().shift(lag.value)
    res = cross_above_rule(ser, SeriesFloat(s_lag))
    return res
def series_cross_below_shift_rule(ser:SeriesFloat, lag:Lag) -> SeriesBool:
    s_lag = ser.to_pd_series().shift(lag.value)
    res = cross_below_rule(ser, SeriesFloat(s_lag))
    return res
#-----------------------------------------------------------------------------------------------------------------------
def series_cross_above_value_rule(ser:SeriesFloat, val:ParameterBase) -> SeriesBool:
    # Generate a constant pandas series
    v_s = pd.Series([val.value] * ser.size)
    res = cross_above_rule(ser, SeriesFloat(v_s))
    return res
def series_cross_below_value_rule(ser:SeriesFloat, val:ParameterBase) -> SeriesBool:
    # Generate a constant pandas series
    v_s = pd.Series([val.value] * ser.size)
    res = cross_below_rule(ser, SeriesFloat(v_s))
    return res
def series_above_value_rule(ser:SeriesFloat, val:ParameterBase) -> SeriesBool:
    # Generate a constant pandas series
    v_s = pd.Series([val.value] * ser.size)
    res = is_above_rule(ser, SeriesFloat(v_s))
    return res
def series_below_value_rule(ser:SeriesFloat, val:ParameterBase) -> SeriesBool:
    # Generate a constant pandas series
    assert val.param_type == "Output", "The parameter value given to series_below_value_rule is not an output type"
    v_s = pd.Series([val.value] * ser.size)
    res = is_below_rule(ser, SeriesFloat(v_s))
    return res
#-----------------------------------------------------------------------------------------------------------------------
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
