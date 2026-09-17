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

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd evoquant

# Install dependencies and set up the project
uv sync
```

### Using pip

```bash
pip install evoquant
```

### Dependencies

Core dependencies:
- `deap` - Distributed Evolutionary Algorithms in Python
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pandas_ta` - Technical analysis indicators
- `nautilus_trader` - High-performance backtesting and trading engine (primary)
- `jinja2` - Template engine for code generation
- `backtesting` - Backtesting engine (legacy support)

Optional dependencies:
- `vectorbt` - Vectorized backtesting (legacy support)
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
│   ├── base.py           # Translator base classes and implementations
│   ├── jinja_template.py # Jinja2 template engine
│   └── ctrader_template.j2 # cTrader C# template (Jinja2)
├── backtest_engine/       # Backtesting components
│   ├── evo_bt.py         # Evolution strategy backtester
│   ├── nautilus_adapter.py # NautilusTrader adapter (primary)
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

EvoQuant can translate evolved strategies to different trading platform languages using Jinja2 templates:

```python
from translator_engine.jinja_template import CTraderJinjaTranslator

# Initialize the translator
translator = CTraderJinjaTranslator()

# Translate an evolved strategy tree to cTrader C# code
csharp_code = translator.translate(evolved_strategy_tree)

# Save the generated code to a file
translator.save_to_file(evolved_strategy_tree, 'output_strategy.cs')
```

The translation engine supports:
- **Jinja2 Templates**: Flexible template system with custom filters
- **Conditional Logic**: Handle None values and optional parameters
- **Type Conversion**: Automatic conversion of Python types to C# syntax
- **Code Formatting**: Proper indentation and structure preservation

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

Built on top of `NautilusTrader` (primary) with legacy support for `backtesting.py` and `vectorbt`:

- **Event-driven**: Simulates realistic order execution
- **High-performance**: Optimized for speed with NautilusTrader
- **Flexible**: Support for various order types, commission models, and exit conditions
- **Comprehensive**: Detailed trade statistics, equity curves, and drawdown analysis
- **Extensible**: Adapter pattern allows easy integration of alternative backtesting engines

```python
from backtest_engine.nautilus_adapter import NautilusBacktestEngine, NautilusMetricsCalculator

# Initialize the backtest engine
engine = NautilusBacktestEngine(
    cash=100_000,
    direction="LongOnly",  # or "ShortOnly", "LongShort"
    commission_rate=0.001,
    slippage_rate=0.001
)

# Run backtest
results = engine.run(data, signals)

# Calculate performance metrics
metrics = NautilusMetricsCalculator()
sharpe = metrics.calculate_sharpe(results.equity_curve)
drawdown = metrics.calculate_max_drawdown(results.equity_curve)
```

## Testing

### Running Tests

Run the test suite using pytest:

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# Or directly with pytest
pytest tests/ -v
```

### CI/CD Pipeline

EvoQuant uses GitHub Actions for continuous integration and deployment. The CI pipeline automatically runs on every push and pull request:

- **Linting**: Code style checks with Ruff
- **Type Checking**: Static type analysis with mypy
- **Testing**: Run all unit and integration tests
- **Coverage**: Generate code coverage reports

The workflow is defined in `.github/workflows/ci-cd.yml` and supports Python 3.10, 3.11, and 3.12.

### Available Tests

- `test_jinja_template.py` - Jinja2 template engine tests
- `test_nautilus_adapter.py` - NautilusTrader backtesting adapter tests
- `test_backtesting.py` - Legacy backtesting engine tests
- `test_indicators.py` - Indicator calculations
- `test_signals.py` - Signal generation
- `test_base.py` - Base class functionality
- `test_utils.py` - Utility functions

## Examples

See the `tests/Example Generated Strategy/` directory for example evolved strategies.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd evoquant
   ```

2. **Install dependencies with uv**:
   ```bash
   uv sync --dev
   ```

3. **Run tests**:
   ```bash
   uv run pytest tests/ -v
   ```

4. **Format code**:
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   ```

5. **Type checking**:
   ```bash
   uv run mypy src/evoquant
   ```

### Code Quality Standards

- Follow PEP 8 style guidelines
- Write unit tests for new features
- Maintain type hints for all public APIs
- Ensure CI/CD pipeline passes before submitting PRs

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
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical analysis library
- [quantstats](https://github.com/ranaroussi/quantstats) - Portfolio analytics (optional)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) - High-performance backtesting and trading engine
