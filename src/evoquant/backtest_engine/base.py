"""
Abstract base classes for backtesting engine abstraction.

This module defines the interface that all backtesting engines must implement,
allowing easy swapping of backtesting implementations (NautilusTrader, backtesting.py, etc.)
without affecting the rest of the codebase.

Follows SOLID principles, particularly:
- Dependency Inversion Principle: High-level modules depend on abstractions
- Interface Segregation Principle: Small, focused interfaces
- Open/Closed Principle: Open for extension, closed for modification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs.

    Attributes:
        initial_cash: Starting capital for backtest
        commission: Commission rate (e.g., 0.001 for 0.1%)
        slippage: Slippage amount or rate
        margin: Margin requirement (1.0 = no margin)
        direction: Trading direction ("LongOnly", "ShortOnly", "LongShort")
        trade_size: Position size (fraction of capital or absolute units)
        stop_loss: Stop-loss configuration tuple (value, unit_type)
        take_profit: Take-profit configuration tuple (value, unit_type)
        exit_after_n_bars: Maximum bars to hold position
        exit_encoded_entry: Whether signal end triggers exit
    """

    initial_cash: float = 100000.0
    commission: float = 0.0001
    slippage: float = 0.0
    margin: float = 1.0
    direction: str = "LongOnly"
    trade_size: float | int = 0.99
    stop_loss: tuple[float, str] | None = None
    take_profit: tuple[float, str] | None = None
    exit_after_n_bars: int | None = None
    exit_encoded_entry: bool = True


@dataclass
class Trade:
    """Represents a single trade.

    Attributes:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Price at entry
        exit_price: Price at exit
        size: Position size (positive for long, negative for short)
        pnl: Profit and loss in currency units
        return_pct: Return percentage
        direction: "Long" or "Short"
        mae: Maximum adverse excursion
        mfe: Maximum favorable excursion
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    direction: str
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class BacktestResult:
    """Standardized backtest result object.

    Attributes:
        trades: List of Trade objects
        equity_curve: DataFrame with equity progression
        returns: Series of period returns
        metrics: Dictionary of performance metrics
        config: BacktestConfig used for this run
        strategy_name: Name of the strategy tested
    """

    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame | None = None
    returns: pd.Series | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    config: BacktestConfig | None = None
    strategy_name: str = ""

    @property
    def is_empty(self) -> bool:
        """Check if result contains no trades."""
        return len(self.trades) == 0

    @property
    def total_return(self) -> float:
        """Get total return from equity curve."""
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        initial = self.equity_curve["Equity"].iloc[0]
        final = self.equity_curve["Equity"].iloc[-1]
        return (final - initial) / initial if initial != 0 else 0.0

    @property
    def n_trades(self) -> int:
        """Get number of trades."""
        return len(self.trades)


class BacktestEngineAdapter(ABC):
    """Abstract base class for all backtesting engines.

    This defines the contract that any backtesting implementation must follow.
    Implementations can use NautilusTrader, backtesting.py, vectorbt, or any
    other backtesting framework.

    Usage:
        class NautilusBacktestEngine(BacktestEngineAdapter):
            def __init__(self, config: BacktestConfig):
                super().__init__(config)
                # Initialize Nautilus-specific components

            def run(self, signals: pd.Series, data: pd.DataFrame) -> BacktestResult:
                # Implement NautilusTrader backtest logic
                pass
    """

    def __init__(self, config: BacktestConfig | None = None):
        """Initialize the backtest engine.

        Args:
            config: Backtest configuration. If None, uses default config.
        """
        self._config = config or BacktestConfig()
        self._last_result: BacktestResult | None = None

    @property
    def config(self) -> BacktestConfig:
        """Get the current backtest configuration."""
        return self._config

    @config.setter
    def config(self, value: BacktestConfig):
        """Set the backtest configuration."""
        self._config = value

    @property
    def last_result(self) -> BacktestResult | None:
        """Get the last backtest result."""
        return self._last_result

    @abstractmethod
    def run(self, signals: pd.Series, data: pd.DataFrame, strategy_name: str = "") -> BacktestResult:
        """Run backtest with given signals and data.

        Args:
            signals: Boolean series indicating entry/exit signals
            data: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
                  Index should be datetime
            strategy_name: Optional name for the strategy

        Returns:
            BacktestResult containing trades, equity curve, and metrics

        Raises:
            BacktestError: If backtest fails
        """
        pass

    @abstractmethod
    def get_metrics(self, result: BacktestResult) -> dict[str, float]:
        """Calculate performance metrics from backtest result.

        Args:
            result: BacktestResult from a run

        Returns:
            Dictionary of metric names to values

        Note:
            Common metrics include:
            - Sharpe ratio
            - Sortino ratio
            - Calmar ratio
            - Max drawdown
            - Total return
            - Win rate
            - Profit factor
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format.

        Args:
            data: OHLCV DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        required_columns = {"Open", "High", "Low", "Close"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        if data.empty:
            raise ValueError("Data cannot be empty")


class MetricsCalculator(ABC):
    """Abstract base class for metrics calculation.

    Allows pluggable metrics calculation regardless of backtesting engine used.
    """

    @abstractmethod
    def calculate(self, returns: pd.Series, trades: list[Trade]) -> dict[str, float]:
        """Calculate performance metrics.

        Args:
            returns: Series of period returns
            trades: List of Trade objects

        Returns:
            Dictionary of metric names to values
        """
        pass


class BacktestError(Exception):
    """Exception raised when backtesting fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


__all__ = [
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "BacktestEngineAdapter",
    "MetricsCalculator",
    "BacktestError",
]
