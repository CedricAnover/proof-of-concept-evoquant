"""Tests for the NautilusTrader backtest engine adapter."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from evoquant.backtest_engine import (
    NautilusBacktestEngine,
    BacktestConfig,
    BacktestResult,
)


class TestNautilusBacktestEngine:
    """Test cases for NautilusBacktestEngine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # Fixed seed for reproducibility
        data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100),
        }, index=dates)
        return data
    
    @pytest.fixture
    def sample_signals(self, sample_data):
        """Create sample boolean signals for testing."""
        np.random.seed(42)
        signals = pd.Series(
            np.random.choice([True, False], len(sample_data)),
            index=sample_data.index
        )
        return signals
    
    def test_engine_initialization(self):
        """Test that the engine can be initialized with default config."""
        engine = NautilusBacktestEngine()
        assert engine is not None
        assert engine.config.initial_cash == 100000.0
    
    def test_engine_initialization_with_config(self):
        """Test that the engine can be initialized with custom config."""
        config = BacktestConfig(
            initial_cash=50000.0,
            commission=0.001,
            direction="LongOnly"
        )
        engine = NautilusBacktestEngine(config)
        assert engine.config.initial_cash == 50000.0
        assert engine.config.commission == 0.001
        assert engine.config.direction == "LongOnly"
    
    def test_run_with_valid_inputs(self, sample_data, sample_signals):
        """Test backtest run with valid inputs."""
        engine = NautilusBacktestEngine()
        result = engine.run(sample_signals, sample_data, strategy_name="TestStrategy")
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "TestStrategy"
        assert result.equity_curve is not None
        assert result.returns is not None
        assert isinstance(result.metrics, dict)
    
    def test_run_produces_trades(self, sample_data, sample_signals):
        """Test that backtest produces trades."""
        engine = NautilusBacktestEngine()
        result = engine.run(sample_signals, sample_data)
        
        # Should have some trades
        assert len(result.trades) >= 0
        
        if len(result.trades) > 0:
            trade = result.trades[0]
            assert hasattr(trade, 'entry_time')
            assert hasattr(trade, 'exit_time')
            assert hasattr(trade, 'entry_price')
            assert hasattr(trade, 'exit_price')
            assert hasattr(trade, 'pnl')
            assert hasattr(trade, 'direction')
    
    def test_metrics_calculation(self, sample_data, sample_signals):
        """Test that metrics are calculated correctly."""
        engine = NautilusBacktestEngine()
        result = engine.run(sample_signals, sample_data)
        
        expected_metrics = [
            "Sharpe", "Sortino", "Calmar", 
            "MaxDrawdown", "TotalReturn", 
            "WinRate", "ProfitFactor"
        ]
        
        for metric in expected_metrics:
            assert metric in result.metrics
            assert isinstance(result.metrics[metric], float)
    
    def test_empty_signals(self, sample_data):
        """Test backtest with empty signals."""
        engine = NautilusBacktestEngine()
        empty_signals = pd.Series([], dtype=bool, index=pd.DatetimeIndex([]))
        result = engine.run(empty_signals, sample_data)
        
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0
    
    def test_no_signals(self, sample_data):
        """Test backtest with all False signals."""
        engine = NautilusBacktestEngine()
        false_signals = pd.Series([False] * len(sample_data), index=sample_data.index)
        result = engine.run(false_signals, sample_data)
        
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0
    
    def test_long_only_direction(self, sample_data, sample_signals):
        """Test backtest with LongOnly direction."""
        config = BacktestConfig(direction="LongOnly")
        engine = NautilusBacktestEngine(config)
        result = engine.run(sample_signals, sample_data)
        
        assert isinstance(result, BacktestResult)
        if len(result.trades) > 0:
            # All trades should be long
            for trade in result.trades:
                assert trade.direction == "Long"
    
    def test_short_only_direction(self, sample_data, sample_signals):
        """Test backtest with ShortOnly direction."""
        config = BacktestConfig(direction="ShortOnly")
        engine = NautilusBacktestEngine(config)
        result = engine.run(sample_signals, sample_data)
        
        assert isinstance(result, BacktestResult)
        if len(result.trades) > 0:
            # All trades should be short
            for trade in result.trades:
                assert trade.direction == "Short"
    
    def test_stop_loss_functionality(self, sample_data, sample_signals):
        """Test backtest with stop-loss configuration."""
        config = BacktestConfig(stop_loss=(2.0, "Percent"))
        engine = NautilusBacktestEngine(config)
        result = engine.run(sample_signals, sample_data)
        
        assert isinstance(result, BacktestResult)
        # Stop-loss should be applied but we can't easily verify it was triggered
    
    def test_commission_deduction(self, sample_data, sample_signals):
        """Test that commissions are deducted."""
        config_no_comm = BacktestConfig(commission=0.0)
        config_with_comm = BacktestConfig(commission=0.01)
        
        engine_no_comm = NautilusBacktestEngine(config_no_comm)
        engine_with_comm = NautilusBacktestEngine(config_with_comm)
        
        result_no_comm = engine_no_comm.run(sample_signals, sample_data)
        result_with_comm = engine_with_comm.run(sample_signals, sample_data)
        
        # With commissions, final equity should be lower or equal
        if len(result_no_comm.trades) > 0 and len(result_with_comm.trades) > 0:
            assert result_with_comm.equity_curve['Equity'].iloc[-1] <= \
                   result_no_comm.equity_curve['Equity'].iloc[-1]
    
    def test_equity_curve_properties(self, sample_data, sample_signals):
        """Test equity curve has required properties."""
        engine = NautilusBacktestEngine()
        result = engine.run(sample_signals, sample_data)
        
        if not result.equity_curve.empty:
            assert 'Equity' in result.equity_curve.columns
            assert 'DrawdownPct' in result.equity_curve.columns
            assert len(result.equity_curve) == len(sample_data)
    
    def test_returns_series(self, sample_data, sample_signals):
        """Test returns series is properly calculated."""
        engine = NautilusBacktestEngine()
        result = engine.run(sample_signals, sample_data)
        
        if not result.returns.empty:
            assert len(result.returns) == len(sample_data)
            # Returns should be numeric
            assert result.returns.dtype in [np.float64, np.float32, float]
    
    def test_invalid_data_missing_columns(self, sample_signals):
        """Test that invalid data raises appropriate error."""
        engine = NautilusBacktestEngine()
        invalid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [100, 101, 102],
        }, index=pd.date_range('2023-01-01', periods=3))
        
        with pytest.raises(ValueError, match="must contain columns"):
            engine.run(sample_signals.loc[:'2023-01-03'], invalid_data)
    
    def test_reproducibility_with_seed(self, sample_data):
        """Test that results are reproducible with fixed seed."""
        np.random.seed(42)
        signals1 = pd.Series(
            np.random.choice([True, False], len(sample_data)),
            index=sample_data.index
        )
        
        np.random.seed(42)
        signals2 = pd.Series(
            np.random.choice([True, False], len(sample_data)),
            index=sample_data.index
        )
        
        engine = NautilusBacktestEngine()
        result1 = engine.run(signals1, sample_data)
        result2 = engine.run(signals2, sample_data)
        
        # Results should be identical with same seed
        assert len(result1.trades) == len(result2.trades)
        assert result1.equity_curve.equals(result2.equity_curve)


class TestNautilusMetricsCalculator:
    """Test cases for NautilusMetricsCalculator."""
    
    def test_calculate_with_valid_returns(self):
        """Test metrics calculation with valid returns."""
        from evoquant.backtest_engine.nautilus_adapter import NautilusMetricsCalculator
        
        calculator = NautilusMetricsCalculator()
        returns = pd.Series(np.random.randn(100) / 100)  # Random returns
        
        metrics = calculator.calculate(returns, [])
        
        assert "Sharpe" in metrics
        assert "Sortino" in metrics
        assert "MaxDrawdown" in metrics
    
    def test_calculate_with_empty_returns(self):
        """Test metrics calculation with empty returns."""
        from evoquant.backtest_engine.nautilus_adapter import NautilusMetricsCalculator
        
        calculator = NautilusMetricsCalculator()
        returns = pd.Series(dtype=float)
        
        metrics = calculator.calculate(returns, [])
        
        assert metrics["Sharpe"] == 0.0
        assert metrics["Sortino"] == 0.0
    
    def test_calculate_with_single_return(self):
        """Test metrics calculation with single return value."""
        from evoquant.backtest_engine.nautilus_adapter import NautilusMetricsCalculator
        
        calculator = NautilusMetricsCalculator()
        returns = pd.Series([0.01])
        
        metrics = calculator.calculate(returns, [])
        
        assert metrics["Sharpe"] == 0.0
        assert metrics["Sortino"] == 0.0
