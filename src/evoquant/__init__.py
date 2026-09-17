"""EvoQuant - Genetic Programming for Trading Strategy Evolution"""

from evoquant.backtest_engine import (
    BacktestConfig,
    NautilusBacktestEngine,
)
from evoquant.base import (
    Lag,
    ParameterBase,
    Period,
    SeriesBase,
    SeriesBool,
    SeriesClose,
    SeriesDate,
    SeriesFloat,
    SeriesHigh,
    SeriesIndicator,
    SeriesLow,
    SeriesOpen,
    SeriesPrice,
    SeriesVolume,
)
from evoquant.indicators import (
    RSI,
    RSIValue,
    ZScore,
    ZScoreValue,
    abs_diff,
    abs_value,
    bbands,
    diff,
    highest,
    lowest,
    rsi,
    shift,
    sma,
    sum_indicator,
    zscore,
)
from evoquant.orchestrator import Evolver, Orchestrator
from evoquant.signals import (
    and2_or1,
    and2_or2,
    and3_or1,
    and3_or2,
    and3_or3,
    and4_or1,
    and_rule,
    and_rule3,
    and_rule4,
    cross_above_rule,
    cross_below_rule,
    is_above_rule,
    is_below_rule,
    not_rule,
    or_rule,
    or_rule3,
    or_rule4,
    series_above_ma_rule,
    series_above_quantile_rule,
    series_above_shift_rule,
    series_above_value_rule,
    series_below_ma_rule,
    series_below_quantile_rule,
    series_below_shift_rule,
    series_below_value_rule,
    series_cross_above_ma_rule,
    series_cross_above_quantile_rule,
    series_cross_above_shift_rule,
    series_cross_below_ma_rule,
    series_cross_below_quantile_rule,
    series_cross_below_shift_rule,
    xor_rule,
    xor_rule3,
    xor_rule4,
)

# Legacy backtesting.py support - only import if available
try:
    from evoquant.backtest_engine import EvoStrategy
except ImportError:
    EvoStrategy = None

from evoquant.translator_engine import CTraderTranslator, PTTranslator

__all__ = [
    # Base types
    "SeriesBase",
    "SeriesFloat",
    "SeriesBool",
    "SeriesDate",
    "SeriesPrice",
    "SeriesIndicator",
    "SeriesOpen",
    "SeriesHigh",
    "SeriesLow",
    "SeriesClose",
    "SeriesVolume",
    "ParameterBase",
    "Period",
    "Lag",
    # Signals
    "and_rule",
    "and_rule3",
    "and_rule4",
    "or_rule",
    "or_rule3",
    "or_rule4",
    "xor_rule",
    "xor_rule3",
    "xor_rule4",
    "not_rule",
    "cross_above_rule",
    "cross_below_rule",
    "is_above_rule",
    "is_below_rule",
    "series_above_value_rule",
    "series_below_value_rule",
    "series_above_shift_rule",
    "series_below_shift_rule",
    "series_cross_above_shift_rule",
    "series_cross_below_shift_rule",
    "series_above_ma_rule",
    "series_below_ma_rule",
    "series_cross_above_ma_rule",
    "series_cross_below_ma_rule",
    "series_above_quantile_rule",
    "series_below_quantile_rule",
    "series_cross_above_quantile_rule",
    "series_cross_below_quantile_rule",
    "and2_or1",
    "and3_or1",
    "and2_or2",
    "and3_or2",
    "and3_or3",
    "and4_or1",
    # Indicators
    "shift",
    "sum_indicator",
    "diff",
    "abs_value",
    "abs_diff",
    "sma",
    "highest",
    "lowest",
    "rsi",
    "zscore",
    "bbands",
    "RSI",
    "ZScore",
    "RSIValue",
    "ZScoreValue",
    # Orchestrator
    "Orchestrator",
    "Evolver",
    # Backtest Engine (NautilusTrader - primary)
    "NautilusBacktestEngine",
    "BacktestConfig",
    # Translator Engine
    "PTTranslator",
    "CTraderTranslator",
]

# Add legacy backtesting support if available
try:
    from evoquant.backtest_engine import EvoStrategy  # type: ignore

    __all__.append("EvoStrategy")
except ImportError:
    pass

__version__ = "0.1.0"
