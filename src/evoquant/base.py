import pandas as pd
import numpy as np
import copy

from typing import Union

class SeriesBase:
    """
    Parameters
    ----------
    in_series : Union[pd.Series, np.ndarray] of any dtype (Required)
    name : Str (Optional) Default=None
    na_value : Any (Optional) Default=np.nan
        Value to replace null. What the null values should be if null.
    in_dtype : Type
        Type of all elements. This is for enforcing what the expected dtype should be.

    Properties
    ----------
    series : np.ndarray
    name : str
    d_type : Any {float, int, bool, str, np.datetime64}
    size : int, Length of series

    Methods
    -------
    to_pd_series() -> pd.Series1

    SeriesBase(df['Close']).name -> 'Close'
    SeriesBase(df['Close'], name='C').name -> 'C'
    """
    def __init__(self, in_series:Union[pd.Series, np.ndarray], name=None, na_value=np.nan):
        # Convert in_series to a np.ndarray. Make sure to deep copy and keep the null values if any and Reset the Index to integer.
        self._size = len(in_series) # Get the length immediately
        if type(in_series) == pd.Series:
            from pandas.api.types import is_datetime64_ns_dtype
            if is_datetime64_ns_dtype(in_series.dtype):
                self._series = copy.deepcopy(in_series.values)
            elif in_series.dtype in [float, int, bool]:
                self._series = copy.deepcopy(in_series.reset_index(drop=True).to_numpy(na_value=na_value))
            elif in_series.dtype in [np.object, str]:
                self._series = copy.deepcopy(in_series.reset_index(drop=True).to_numpy(na_value=np.nan))
            else:
                raise "Cannot convert the input series to a numpy ndarray!"
        elif type(in_series) == np.ndarray:
            # WARN: By default, Numpy would treat the dtype as float (and can accept dtype int & bool)!
            self._series = copy.deepcopy(in_series)
            if self._series.dtype in [float, int, bool]:
                self._series[np.isnan(self._series)] = na_value # This only work for float or int
                self._series = np.asarray(self._series)
            elif self._series.dtype == np.object:
                raise TypeError("If in_series is np.ndarray, SeriesBase can only handle dtype of float. :(")
        else:
            raise TypeError("The type of in_series must be either pandas.Series or numpy.ndarray!")

        self._d_type = self._series.dtype

        self._name = name

    @property
    def series(self) -> np.ndarray:
        return self._series

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        assert isinstance(new_name, str)
        self._name = new_name

    @property
    def d_type(self):
        return self._d_type

    @property
    def size(self):
        if self._size != self._series.shape[0]:
            raise ValueError("The size of the original input arg is not the same as the self.series np.ndarray!")
        else:
            return self._series.shape[0] #self._size

    def to_pd_series(self, *args, **kwargs) -> pd.Series:
        """Convert & Return the pd.Series of self.series"""
        return copy.deepcopy(pd.Series(self._series, name=self._name, *args, **kwargs))

class SeriesFloat(SeriesBase):
    """"""
    def __init__(self, in_series, name=None, na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)
        # Check if the dtypes in have at least 1 float and no other dtypes in series. Otherwise, return an error.
        if self._d_type != float:
            raise TypeError("D-type of the given series has to be float.")

        assert self._d_type == float

        # Make Sure to Convert the numpy array as a float. Make a Deep Copy.
        self._series = copy.deepcopy(self._series.astype('float', casting='unsafe'))
        self._d_type = self._series.dtype

    @classmethod
    def is_stationary(self):
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(self._series)
        p_value = result[1]
        return True if p_value <= 0.05 else False

class SeriesIndicator(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Indicator", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesPrice(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Price", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesOpen(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Open", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesHigh(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="High", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesLow(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Low", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesClose(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Close", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesVolume(SeriesFloat, SeriesBase):
    def __init__(self, in_series, name="Volume", na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)

class SeriesBool(SeriesBase):
    def __init__(self, in_series, name=None, na_value=False):
        super().__init__(in_series, name=name, na_value=na_value)
        if self._d_type not in [bool]:
            # Need to make sure that if dtype is object, it should only contain bool or np.nan
            if np.all(np.logical_or(self._series == True, self._series == False, self._series==np.nan)):
                self._series = copy.deepcopy(self._series.astype('bool', casting='unsafe'))
                self._d_type = self._series.dtype
                assert self._d_type == bool
            else:
                raise TypeError("The Series contain values that are neither bool or np.nan")

        self._bool_as_int = self.__bool_as_int(self._series)

    @property
    def bool_as_int(self):
        return self._bool_as_int

    def get_signal(self, signal_mode='long', type_mode='int'):
        """Returns a np.ndarray that contain values {1,0,-1}
        signal_mode : Str {'long', 'short', 'longshort'} (Optional) Default='long'
        type_mode : Str {'int', 'str'} (Optional) Default='int'
        """
        def out_signals(x, signal_mode=signal_mode, type_mode=type_mode):
            if type_mode == 'int':
                if signal_mode == 'long': return 1 if x==True else 0
                if signal_mode == 'short': return -1 if x==True else 0
                if signal_mode == 'longshort': return 1 if x==True else -1
            if type_mode == 'str':
                if signal_mode == 'long': return "Buy" if x==True else "Flat"
                if signal_mode == 'short': return "Sell" if x==True else "Flat"
                if signal_mode == 'longshort': return "Buy" if x==True else "Sell"
        applyall = np.vectorize(out_signals)
        return applyall(self._series, signal_mode=signal_mode, type_mode=type_mode) # np.ndarray

    def __bool_as_int(self, in_arr): #--> np.ndarray<int> {1,0}
        applyall = np.vectorize(lambda x: 1 if x else 0, otypes='i')
        return applyall(in_arr) # np.ndarray<int> {1,0}

class SeriesDate(SeriesBase):
    """
    Properties
    ----------
    day_of_week : np.ndarray<int>
    month_in_year : np.ndarray<int>
    """
    def __init__(self, in_series, name='Date', na_value=np.nan):
        super().__init__(in_series, name=name, na_value=na_value)
        # Very likely that the dtype for a date or datetime index would be string or object
        # Convert to datetime64
        self._series = np.array(list(in_series), dtype='datetime64')
        self._d_type = self._series.dtype
        self._day_of_week = self.__day_of_week(self._series)
        self._month_in_year = self.__month_in_year(self._series)
        self._hour_in_day = self.__hour_in_day(self._series)

    @property
    def day_of_week(self):
        return self._day_of_week

    @property
    def month_in_year(self):
        return self._month_in_year

    @property
    def hour_in_day(self):
        return self._hour_in_day

    def __day_of_week(self, in_array):
        """Return the day of week
        Note: We are using pandas for ease of calculation.
        Note: We are not applying any offset/lag/shift. But this will be important in backtesting.
        1=Mon,...,7=Sun
        """
        #out = (self._series - np.timedelta64(1, 'D')).astype('datetime64[W]').view('int') % 7 + 1
        return pd.Series(in_array, name="DayOfWeek").dt.dayofweek.add(1).values # np.ndarray<int>

    def __month_in_year(self, in_array):
        # 1=Jan,...,12=Dec
        return pd.Series(in_array, name="MonthInYear").dt.month.values # np.ndarray<int>

    def __hour_in_day(self, in_array):
        # 0=12am,...,12=12pm,13=1pm,...,23=11pm
        return pd.Series(in_array, name="HourInDay").dt.hour.values


class ParameterBase:
    """
    These are parameters to be used as Terminals or Ephemirals in DEAP.
    This class is referring to the parameters in indicators or the values of indicators.
    Typically, parameters are int, float, str, bool
    Subclasses of this Class can add additional layer of Type & Value Constraints.

    Note that ParameterBase can also be used in a signal/rule, especially when comparing an indicator
    to a typed value.

    Example: String Styles
    1 - Period=1
    2 - Period1
    3 - Period(1)
    4 - Period_1
    5 - 1

    Example: value_mapper
    obj = DayOfWeek(1)
    obj.value_mapper(trade_platform="ctrader") -> "DayOfWeek.Monday" //A Valid String for cTrader Code
    """

    def __init__(self, in_value, param_name, param_type, str_style=1):
        _param_types = ["Input", "Output"] # The type of parameter for terminals. Because it can be an input parameter or output value for an indicator or signal.
        assert isinstance(str_style, int) and isinstance(param_name, str), "str_style must be an integer and param_name must be a string"
        assert str_style in [1, 2, 3, 4, 5], "String Styles can only be an integer from 1 to 5"
        assert isinstance(param_type, str) and param_type in _param_types, f"param_type must be valid. Choose from {_param_types}."

        self._value = in_value
        self._d_type = type(in_value)
        self._param_name = param_name
        self._str_style = str_style
        self._param_type = param_type

    @property
    def value(self):
        return self._value
    @property
    def d_type(self):
        return self._d_type
    @property
    def param_name(self):
        return self._param_name
    @property
    def param_type(self):
        return self._param_type

    @property
    def str_style(self):
        return self._str_style

    @param_name.setter
    def param_name(self, new_param_name):
        assert isinstance(new_param_name, str)
        self._param_name = new_param_name

    @str_style.setter
    def str_style(self, new_str_style):
        assert isinstance(new_str_style, int)
        self._str_style = new_str_style

    def value_mapper(self, trade_platform):
        """This will be implemented to Subclasses that returns string values in order to map to valid synax in a target programming language"""
        # This will be invoked in the 'translation' of PrameterBase value to a target language (eg C# cTrader)
        return str(self._value) # This is the default implementation. Needs to be overriden, specially if the d_type is string.

    def __str__(self):
        if self._str_style == 1:
            return f"{self._param_name}={self._value}"
        if self._str_style == 2:
            return f"{self._param_name}{self._value}"
        if self._str_style == 3:
            return f"{self._param_name}({self._value})"
        if self._str_style == 4:
            return f"{self._param_name}_{self._value}"
        if self._str_style == 5:
            return f"{self._value}"

    def __repr__(self):
        if isinstance(self._value, str): # For Strings
            return r"""{0}('{1}')""".format(self.__class__.__name__, self._value) # Text must be raw and valid python executable string code
        elif isinstance(self._value, (int, float)): # For Numeric
            return f"{self.__class__.__name__}({self._value})"
        else: # For other like pandas/numpy object or date or datetime, we try similar to Numeric
            return f"{self.__class__.__name__}({self._value})"

class Period(ParameterBase):
    # def __init__(self, in_value, param_name="Period", param_type="Input", str_style=1):
    def __init__(self, in_value, str_style=1):
        super().__init__(in_value, param_name="Period", param_type="Input", str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be an integer.")
        if in_value < 5:
            raise ValueError(f"Period must be at least 5.")

class Lag(ParameterBase):
    def __init__(self, in_value, param_name="Lag", param_type="Input", str_style=1):
        super().__init__(in_value, param_name=param_name, param_type=param_type, str_style=str_style)
        if not isinstance(in_value, int):
            raise TypeError("Value must be an integer.")
        if in_value < 1:
            raise ValueError(f"Lag must be at least 1.")

__all__ = ["SeriesBase", "SeriesFloat", "SeriesBool", "SeriesDate", "SeriesPrice", "SeriesIndicator",
           "SeriesOpen", "SeriesHigh", "SeriesLow", "SeriesClose", "SeriesVolume",
           "ParameterBase", "Period", "Lag"]

