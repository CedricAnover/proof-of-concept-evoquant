# EvoQuant

**EvoQuant** is a genetic programming (GP) framework for automatically evolving and optimizing algorithmic trading strategies using DEAP (Distributed Evolutionary Algorithms in Python).

## Overview

EvoQuant combines evolutionary computation with quantitative finance to discover profitable trading strategies. It uses genetic programming to evolve trading rules represented as expression trees, which are then backtested and evaluated using performance metrics.

## Features

- **Genetic Programming Engine**: Evolves trading strategies using DEAP's GP framework
- **Backtesting Integration**: Built-in support for backtesting strategies on historical OHLCV data
- **Technical Indicators**: Support for common indicators (SMA, RSI, Bollinger Bands, Z-Score, etc.)
- **Signal Generation**: Boolean signal generation with logical operators (AND, OR, XOR, NOT)
- **Multi-Platform Translation**: Code translation capabilities for different trading platforms (cTrader, MetaTrader, etc.)
- **Performance Metrics**: Comprehensive fitness evaluation using Sharpe ratio, Sortino ratio, and other quantstats/empyrical metrics
- **Parallel Processing**: Support for multiprocessing to speed up evolution
- **Custom Type System**: Strongly-typed GP with custom series types (SeriesFloat, SeriesBool, SeriesDate, etc.)

## Installation

```bash
pip install evoquant
```

### Dependencies

- `deap` - Distributed Evolutionary Algorithms in Python
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pandas_ta` - Technical analysis indicators
- `backtesting` - Backtesting engine
- `vectorbt` - Vectorized backtesting
- `quantstats` - Portfolio analytics
- `empyrical` - Financial risk metrics
- `statsmodels` - Statistical tests

## Project Structure

```
evoquant/
├── base.py                 # Base classes for Series and Parameters
├── evo_gp.py              # Genetic programming engine and utilities
├── indicators.py          # Technical indicators (RSI, BBands, ZScore, etc.)
├── signals.py             # Signal generation rules and logic operators
├── orchestrator.py        # Main orchestration classes (Orchestrator, Evolver)
├── translator_engine/     # Code translation engines
│   ├── base.py
│   └── ctrader_template.cs
├── backtest_engine/       # Backtesting components
│   ├── evo_bt.py         # Evolution strategy backtester
│   ├── utils.py          # Backtesting utilities
│   └── validation.py     # Validation logic
└── tests/                # Test suite
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from evoquant import Orchestrator, SeriesClose, SeriesOpen, SeriesHigh, SeriesLow
from evoquant.indicators import sma, rsi
from evoquant.base import Period, Lag

# Load your OHLCV data
df = pd.read_csv('your_data.csv')  # Must have columns: Date, Open, High, Low, Close, Volume

# Initialize the orchestrator
orchestrator = Orchestrator(
    symbol='SPY',
    timeframe='1d',
    data=df
)

# Configure and run evolution
orchestrator.run()

# Get the optimal strategy
optimal_strategy = orchestrator.get_optimal_strategy()
```

### Core Components

#### Series Types

EvoQuant provides type-safe series wrappers:

```python
from evoquant.base import SeriesClose, SeriesOpen, SeriesHigh, SeriesLow, SeriesVolume, SeriesBool, SeriesDate

close = SeriesClose(df['Close'])
open_price = SeriesOpen(df['Open'])
high = SeriesHigh(df['High'])
low = SeriesLow(df['Low'])
volume = SeriesVolume(df['Volume'])
dates = SeriesDate(df['Date'])
```

#### Parameters

```python
from evoquant.base import Period, Lag

period = Period(14)  # Must be >= 5
lag = Lag(3)         # Must be >= 1
```

#### Indicators

```python
from evoquant.indicators import sma, rsi, bbands, zscore, highest, lowest
from evoquant.base import Period, StdDev, MAMode, BBOut

# Simple Moving Average
sma_14 = sma(close, Period(14))

# RSI
rsi_14 = rsi(close, Period(14))

# Bollinger Bands
bb_upper = bbands(close, Period(20), StdDev(2.0), MAMode('sma'), BBOut('bbu'))

# Z-Score
zscore_20 = zscore(close, Period(20))
```

#### Signals

```python
from evoquant.signals import (
    cross_above_rule, cross_below_rule, 
    is_above_rule, is_below_rule,
    and_rule, or_rule, not_rule, xor_rule
)

# Crossover signals
bullish_cross = cross_above_rule(sma_10, sma_20)
bearish_cross = cross_below_rule(sma_10, sma_20)

# Logical combinations
signal = and_rule(bullish_cross, is_above_rule(rsi_14, rsi_threshold))
```

### Backtesting Configuration

```python
from backtest_engine.evo_bt import EvoStrategy

# Configure strategy parameters
EvoStrategy.direction = "LongOnly"  # or "ShortOnly", "LongShort"
EvoStrategy.trade_size = 0.99  # 99% of available capital
EvoStrategy.stop_loss = (2.0, 'Percent')  # 2% stop loss
EvoStrategy.exit_after_n_bars = 10  # Exit after 10 bars
```

### Fitness Functions

EvoQuant supports multiple fitness functions for strategy evaluation:

- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside deviation adjusted returns
- **Calmar Ratio** - Return relative to maximum drawdown
- **Max Drawdown** - Maximum peak-to-trough decline
- **Total Return** - Cumulative return
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss

```python
from evoquant.evo_gp import PerfStats

# Define fitness functions with weights
perf_stats = PerfStats([
    ("Sharpe", 1.0),      # Maximize Sharpe ratio
    ("Sortino", 0.5),     # Maximize Sortino ratio
    ("MaxDrawdown", -1.0) # Minimize max drawdown
])
```

## Advanced Usage

### Custom Indicators

```python
from evoquant.base import SeriesFloat, Period
import pandas_ta as ta

def custom_indicator(series: SeriesFloat, period: Period) -> SeriesFloat:
    """Create a custom technical indicator"""
    result = ta.your_custom_function(series.to_pd_series(), length=period.value)
    return SeriesFloat(result, name="CustomIndicator")
```

### Custom Signals

```python
from evoquant.base import SeriesBool, SeriesFloat

def custom_signal_rule(ser1: SeriesFloat, ser2: SeriesFloat) -> SeriesBool:
    """Define a custom signal rule"""
    result = ser1.series > (ser2.series * 1.5)
    return SeriesBool(result, name="CustomSignal")
```

### Code Translation

EvoQuant can translate evolved strategies to different trading platform languages:

```python
from translator_engine.base import CTraderTranslator

translator = CTraderTranslator()
csharp_code = translator.translate(evolved_strategy_tree)
```

## Architecture

### Type System

EvoQuant uses a strongly-typed GP system:

- **SeriesBase** - Base class for all series types
  - **SeriesFloat** - Float-valued series (prices, indicators)
  - **SeriesBool** - Boolean series (signals)
  - **SeriesDate** - Date/datetime series
  - **SeriesPrice** - Price series (Open, High, Low, Close)
  - **SeriesIndicator** - Indicator output series

- **ParameterBase** - Base class for parameters
  - **Period** - Lookback period (int >= 5)
  - **Lag** - Shift/lag value (int >= 1)
  - **StdDev** - Standard deviation multiplier (float > 0)
  - **MAMode** - Moving average mode ('ema', 'sma')
  - **BBOut** - Bollinger Bands output ('bbl', 'bbm', 'bbu', 'bbb', 'bbp')

### Genetic Programming

The GP engine evolves strategies as expression trees:

1. **Initialization**: Generate random population of strategy trees
2. **Evaluation**: Backtest each strategy and calculate fitness
3. **Selection**: Select best performing strategies
4. **Crossover**: Combine parts of parent strategies
5. **Mutation**: Randomly modify strategy trees
6. **Repeat**: Continue until exit criteria met

### Backtesting Engine

Built on top of `backtesting.py` and `vectorbt`:

- **Event-driven**: Simulates realistic order execution
- **Vectorized**: Fast portfolio-level calculations
- **Flexible**: Support for various order types and exit conditions
- **Comprehensive**: Detailed trade statistics and equity curves

## Testing

Run the test suite:

```bash
cd tests
python -m pytest
```

Available tests:
- `test_backtesting.py` - Backtesting engine tests
- `test_indicators.py` - Indicator calculations
- `test_signals.py` - Signal generation
- `test_base.py` - Base class functionality
- `test_utils.py` - Utility functions

## Examples

See the `tests/Example Generated Strategy/` directory for example evolved strategies.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[Add your license here]

## Citation

If you use EvoQuant in your research, please cite:

```bibtex
@software{evoquant,
  title = {EvoQuant: Genetic Programming for Trading Strategy Evolution},
  year = {2024}
}
```

## Acknowledgments

- [DEAP](https://github.com/DEAP/deap) - Distributed Evolutionary Algorithms in Python
- [backtesting.py](https://github.com/kernc/backtesting.py) - Backtesting framework
- [vectorbt](https://github.com/polakowo/vectorbt) - Vectorized backtesting
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis library
- [quantstats](https://github.com/ranaroussi/quantstats) - Portfolio analytics
