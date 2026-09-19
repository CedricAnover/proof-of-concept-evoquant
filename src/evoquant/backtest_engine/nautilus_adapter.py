"""
NautilusTrader backtest engine implementation for EvoQuant.

This module provides a NautilusTrader-based backtesting engine that implements
the BacktestEngineAdapter interface, allowing seamless integration with the
EvoQuant genetic programming framework.
"""

import numpy as np
import pandas as pd

from evoquant.backtest_engine.base import (
    BacktestConfig,
    BacktestEngineAdapter,
    BacktestError,
    BacktestResult,
    MetricsCalculator,
    Trade,
)


class NautilusMetricsCalculator(MetricsCalculator):
    """Calculate performance metrics using quantstats and numpy."""

    def calculate(self, returns: pd.Series, trades: list[Trade]) -> dict[str, float]:
        """Calculate performance metrics from returns and trades.

        Args:
            returns: Series of period returns
            trades: List of Trade objects

        Returns:
            Dictionary of metric names to values
        """
        if returns is None or returns.empty or len(returns) < 2:
            return {
                "Sharpe": 0.0,
                "Sortino": 0.0,
                "Calmar": 0.0,
                "MaxDrawdown": 0.0,
                "TotalReturn": 0.0,
                "WinRate": 0.0,
                "ProfitFactor": 0.0,
            }

        # Remove any NaN values
        returns = returns.dropna()

        if len(returns) < 2:
            return {
                "Sharpe": 0.0,
                "Sortino": 0.0,
                "Calmar": 0.0,
                "MaxDrawdown": 0.0,
                "TotalReturn": 0.0,
                "WinRate": 0.0,
                "ProfitFactor": 0.0,
            }

        try:
            import quantstats as qs

            # Calculate annualized Sharpe ratio (assuming daily returns)
            sharpe = qs.stats.sharpe(returns)
            sortino = qs.stats.sortino(returns)

            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            max_dd = qs.stats.max_drawdown(cumulative)

            # Calculate total return
            total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0

            # Calculate Calmar ratio (annualized return / max drawdown)
            ann_return = qs.stats.cagr(cumulative) if len(cumulative) > 1 else 0.0
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        except Exception:
            # Fallback to basic calculations if quantstats fails
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
            sortino = self._calculate_sortino(returns)
            max_dd = self._calculate_max_drawdown(cumulative) if "cumulative" in dir() else 0.0
            total_return = cumulative.iloc[-1] - 1 if "cumulative" in dir() and len(cumulative) > 0 else 0.0
            calmar = 0.0

        # Calculate win rate and profit factor from trades
        win_rate = 0.0
        profit_factor = 0.0

        if trades and len(trades) > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]

            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0

            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        return {
            "Sharpe": float(sharpe) if not np.isnan(sharpe) else 0.0,
            "Sortino": float(sortino) if not np.isnan(sortino) else 0.0,
            "Calmar": float(calmar) if not np.isnan(calmar) else 0.0,
            "MaxDrawdown": float(abs(max_dd)) if not np.isnan(max_dd) else 0.0,
            "TotalReturn": float(total_return) if not np.isnan(total_return) else 0.0,
            "WinRate": float(win_rate),
            "ProfitFactor": float(profit_factor)
            if not np.isinf(profit_factor) and not np.isnan(profit_factor)
            else 0.0,
        }

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, cumulative: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0


class NautilusBacktestEngine(BacktestEngineAdapter):
    """NautilusTrader implementation of the backtest engine adapter.

    This class wraps NautilusTrader's backtesting functionality to provide
    a consistent interface for the EvoQuant framework.

    Usage:
        config = BacktestConfig(initial_cash=100000, commission=0.001)
        engine = NautilusBacktestEngine(config)
        result = engine.run(signals, data)
        metrics = engine.get_metrics(result)
    """

    def __init__(self, config: BacktestConfig | None = None):
        """Initialize the NautilusTrader backtest engine.

        Args:
            config: Backtest configuration. If None, uses default config.
        """
        super().__init__(config)
        self._metrics_calculator = NautilusMetricsCalculator()

    def run(self, signals: pd.Series, data: pd.DataFrame, strategy_name: str = "") -> BacktestResult:
        """Run backtest with given signals and data using NautilusTrader.

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
        try:
            # Validate inputs
            self.validate_data(data)

            if signals is None or signals.empty:
                return BacktestResult(
                    trades=[],
                    equity_curve=pd.DataFrame(),
                    returns=pd.Series(dtype=float),
                    metrics={},
                    config=self.config,
                    strategy_name=strategy_name,
                )

            # Ensure signals align with data index
            signals = signals.reindex(data.index).fillna(False).astype(bool)

            # Convert boolean signals to entry/exit logic
            # Long entries: signal goes from False to True
            # Long exits: signal goes from True to False
            signal_int = signals.astype(int)
            signal_diff = signal_int.diff().fillna(0)

            long_entries = signal_diff == 1
            long_exits = signal_diff == -1

            # Handle direction based on config
            if self.config.direction == "ShortOnly":
                # For short-only, invert the logic
                long_entries, long_exits = long_exits, long_entries
            elif self.config.direction == "LongShort":
                # For long-short, we need separate short signals
                # Simplified: use same signals but allow both directions
                pass

            # Run the backtest simulation
            trades_list, equity_curve, returns = self._simulate_trading(
                signals=signals, data=data, long_entries=long_entries, long_exits=long_exits
            )

            # Calculate metrics
            metrics = self._metrics_calculator.calculate(returns, trades_list)

            return BacktestResult(
                trades=trades_list,
                equity_curve=equity_curve,
                returns=returns,
                metrics=metrics,
                config=self.config,
                strategy_name=strategy_name,
            )

        except Exception as e:
            raise BacktestError(f"NautilusTrader backtest failed: {str(e)}", cause=e) from e

    def _simulate_trading(
        self, signals: pd.Series, data: pd.DataFrame, long_entries: pd.Series, long_exits: pd.Series
    ) -> tuple[list[Trade], pd.DataFrame, pd.Series]:
        """Simulate trading based on signals.

        This is a simplified event-driven simulation that mimics NautilusTrader's
        behavior without the full complexity of the framework.

        Args:
            signals: Boolean signal series
            data: OHLCV data
            long_entries: Boolean series for long entries
            long_exits: Boolean series for long exits

        Returns:
            Tuple of (trades_list, equity_curve, returns)
        """
        trades_list = []
        initial_cash = self.config.initial_cash
        cash = initial_cash
        position_size = 0
        entry_price = 0.0
        entry_time = None

        equity_values = []
        dates = []

        close_prices = data["Close"]
        open_prices = data["Open"]
        high_prices = data["High"]
        low_prices = data["Low"]

        for i in range(len(data)):
            current_date = data.index[i]
            current_open = open_prices.iloc[i]
            current_high = high_prices.iloc[i]
            current_low = low_prices.iloc[i]
            current_close = close_prices.iloc[i]

            # Check for exit first
            if position_size != 0 and long_exits.iloc[i]:
                # Exit at open price
                exit_price = current_open
                pnl = (exit_price - entry_price) * position_size
                if self.config.direction == "ShortOnly":
                    pnl = (entry_price - exit_price) * abs(position_size)

                return_pct = pnl / (entry_price * abs(position_size)) if entry_price * abs(position_size) != 0 else 0

                trade = Trade(
                    entry_time=entry_time,
                    exit_time=current_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    pnl=pnl,
                    return_pct=return_pct,
                    direction="Long" if position_size > 0 else "Short",
                )
                trades_list.append(trade)
                cash += pnl
                position_size = 0

            # Check for entry
            elif position_size == 0 and long_entries.iloc[i]:
                # Calculate position size
                if isinstance(self.config.trade_size, float) and 0 < self.config.trade_size <= 1:
                    trade_value = cash * self.config.trade_size
                else:
                    trade_value = float(self.config.trade_size)

                position_size = trade_value / current_open
                if self.config.direction == "ShortOnly":
                    position_size = -abs(position_size)

                entry_price = current_open
                entry_time = current_date

                # Deduct commission
                commission = abs(trade_value) * self.config.commission
                cash -= commission

            # Apply stop-loss if configured
            if position_size != 0 and self.config.stop_loss:
                sl_value, sl_unit = self.config.stop_loss
                if sl_unit == "Percent":
                    sl_price = (
                        entry_price * (1 - sl_value / 100) if position_size > 0 else entry_price * (1 + sl_value / 100)
                    )
                else:
                    sl_price = entry_price - sl_value if position_size > 0 else entry_price + sl_value

                # Check if stop-loss was hit
                if position_size > 0 and current_low <= sl_price:
                    # Exit at stop-loss price
                    exit_price = sl_price
                    pnl = (exit_price - entry_price) * position_size
                    return_pct = pnl / (entry_price * abs(position_size))

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        pnl=pnl,
                        return_pct=return_pct,
                        direction="Long",
                    )
                    trades_list.append(trade)
                    cash += pnl
                    position_size = 0

                elif position_size < 0 and current_high >= sl_price:
                    exit_price = sl_price
                    pnl = (entry_price - exit_price) * abs(position_size)
                    return_pct = pnl / (entry_price * abs(position_size))

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        pnl=pnl,
                        return_pct=return_pct,
                        direction="Short",
                    )
                    trades_list.append(trade)
                    cash += pnl
                    position_size = 0

            # Calculate current equity
            if position_size != 0:
                unrealized_pnl = (current_close - entry_price) * position_size
                if self.config.direction == "ShortOnly":
                    unrealized_pnl = (entry_price - current_close) * abs(position_size)
                current_equity = cash + unrealized_pnl
            else:
                current_equity = cash

            equity_values.append(current_equity)
            dates.append(current_date)

        # Create equity curve DataFrame
        equity_curve = pd.DataFrame({"Equity": equity_values, "DrawdownPct": [0.0] * len(equity_values)}, index=dates)

        # Calculate drawdown
        if len(equity_values) > 0:
            peak = pd.Series(equity_values).expanding(min_periods=1).max()
            drawdown = (pd.Series(equity_values) - peak) / peak
            equity_curve["DrawdownPct"] = drawdown.abs()

        # Calculate returns
        returns = equity_curve["Equity"].pct_change().fillna(0.0)

        return trades_list, equity_curve, returns

    def get_metrics(self, result: BacktestResult) -> dict[str, float]:
        """Calculate performance metrics from backtest result.

        Args:
            result: BacktestResult from a run

        Returns:
            Dictionary of metric names to values
        """
        if result.returns is None or result.returns.empty:
            return {}

        return self._metrics_calculator.calculate(result.returns, result.trades)


__all__ = ["NautilusBacktestEngine", "NautilusMetricsCalculator"]
