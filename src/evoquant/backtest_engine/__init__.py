"""Backtest Engine for EvoQuant

This module provides backtesting functionality for the EvoQuant framework.
The primary backtesting engine is NautilusTrader, with legacy support for
backtesting.py available through optional dependencies.
"""

from evoquant.backtest_engine.base import (
    BacktestConfig,
    BacktestEngineAdapter,
    BacktestError,
    BacktestResult,
    MetricsCalculator,
    Trade,
)
from evoquant.backtest_engine.nautilus_adapter import (
    NautilusBacktestEngine,
    NautilusMetricsCalculator,
)

# Legacy backtesting.py adapter - only import if backtesting is installed
try:
    from evoquant.backtest_engine.evo_bt import EvoStrategy

    _HAS_LEGACY = True
except ImportError:
    _HAS_LEGACY = False
    EvoStrategy = None

__all__ = [
    # Base classes
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "BacktestEngineAdapter",
    "MetricsCalculator",
    "BacktestError",
    # NautilusTrader implementation (primary)
    "NautilusBacktestEngine",
    "NautilusMetricsCalculator",
]

# Only expose legacy classes if available
if _HAS_LEGACY:
    __all__.append("EvoStrategy")
