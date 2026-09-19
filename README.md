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
- **Performance Metrics**: Comprehensive fitness evaluation using Sharpe ratio, Sortino ratio, and other quantstats metrics
- **Parallel Processing**: Support for multiprocessing to speed up evolution
- **Custom Type System**: Strongly-typed GP with custom series types (SeriesFloat, SeriesBool, SeriesDate, etc.)

## Installation

Requires **Python >= 3.12** and [`uv`](https://docs.astral.sh/uv/).

```bash
# Core + dev dependencies
uv sync --all-extras --dev

# Add legacy backtesting engines (backtesting.py, vectorbt)
uv sync --all-extras --dev --extra legacy
```

### Dependencies

Core dependencies (`pyproject.toml`):
- `deap` - Distributed Evolutionary Algorithms in Python
- `pandas` - Data manipulation
- `numpy` - Numerical computing (< 2.4.0)
- `pandas-ta` - Technical analysis indicators
- `nautilus-trader` - Primary high-performance backtesting and trading engine
- `quantstats` - Portfolio analytics
- `statsmodels` - Statistical tests
- `scikit-learn` - Train/test splitting for validation
- `numba` - Performance optimization
- `jinja2` - Template engine for code generation

Legacy backtesting engines (`uv sync --extra legacy`, optional, for migration):
- `backtesting` - backtesting.py engine (legacy support)
- `vectorbt` - Vectorized backtesting (legacy support)
- `plotly<7.0.0` - Pinned below 7.x because vectorbt 1.0.0 uses `scattermapbox`, removed in plotly 7.x

Dev dependencies:
- `pytest`, `pytest-cov` - Testing
- `ruff` - Linting and formatting
- `mypy` - Type checking

> `empyrical` is not a declared dependency. `evo_gp.py` imports it behind a guarded
> `try/except ImportError` and falls back to `quantstats` when it is absent.

## Project Structure

```
src/evoquant/
├── __init__.py             # Public API exports
├── base.py                 # Series/Parameter type system
├── evo_gp.py              # Genetic programming engine (still legacy-coupled)
├── indicators.py          # Technical indicators (RSI, BBands, ZScore, etc.)
├── signals.py             # Boolean signal rules
├── orchestrator.py        # Orchestrator/Evolver (under construction stubs)
├── translator_engine/     # Code translation engines
│   ├── base.py            # PTTranslator, CTraderTranslator (string.Template)
│   ├── jinja_template.py  # TemplateEngine, CTraderJinjaTranslator (Jinja2)
│   ├── ctrader_template.j2  # cTrader Jinja2 template
│   └── ctrader_template.cs  # Example generated C# output
├── backtest_engine/       # Backtesting components
│   ├── __init__.py
│   ├── base.py              # BacktestConfig, BacktestEngineAdapter, BacktestResult, Trade, MetricsCalculator, BacktestError
│   ├── nautilus_adapter.py  # NautilusTrader implementation (primary engine)
│   ├── evo_bt.py         # Legacy backtesting.py adapter (requires `legacy` extra)
│   ├── utils.py          # Backtesting utilities
│   └── validation.py     # Train/test and walk-forward split logic
└── tests/                  # Test suite (repo root)
```

## Quick Start

> `Orchestrator` and `Evolver` in `evoquant/orchestrator.py` are currently
> under-construction stubs with no working methods. Until they are implemented,
> use the building blocks below directly (series → indicators → signals →
> backtest → translate).

### Basic Usage

```python
import pandas as pd
from evoquant import SeriesClose, SeriesOpen, SeriesHigh, SeriesLow
from evoquant.indicators import sma, rsi
from evoquant.base import Period, Lag

# Load your OHLCV data (columns: Date, Open, High, Low, Close, Volume optional)
df = pd.read_csv('your_data.csv')

close = SeriesClose(df['Close'])
open_price = SeriesOpen(df['Open'])

# Indicators
sma_10 = sma(close, Period(10))
sma_20 = sma(close, Period(20))
rsi_14 = rsi(close, Period(14))

# Signals
from evoquant.signals import cross_above_rule, and_rule, is_above_rule
from evoquant.base import SeriesFloat

bullish_cross = cross_above_rule(sma_10, sma_20)
# is_above_rule compares two SeriesFloat; wrap a threshold the same way
signal = and_rule(bullish_cross, is_above_rule(rsi_14, SeriesFloat(rsi_14.series * 0 + 30)))
```

### Core Components

#### Indicators

```python
from evoquant.indicators import sma, rsi, bbands, zscore, highest, lowest
from evoquant.base import Period
from evoquant.indicators import StdDev, MAMode, BBOut

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

# Logical combinations (all operands are SeriesBool)
from evoquant.base import SeriesFloat
rsi_threshold = SeriesFloat(rsi_14.series * 0 + 30)
signal = and_rule(bullish_cross, is_above_rule(rsi_14, rsi_threshold))
```

### Backtesting Configuration

NautilusTrader is the primary engine. Configure a run with `BacktestConfig`
and execute it with `NautilusBacktestEngine` (see
`tests/test_nautilus_adapter.py` for a full working example):

```python
from evoquant.backtest_engine import BacktestConfig, NautilusBacktestEngine

config = BacktestConfig(
    initial_cash=100000.0,
    direction="LongOnly",  # or "ShortOnly", "LongShort"
    trade_size=0.99,  # 99% of available capital
    stop_loss=(2.0, "Percent"),  # 2% stop loss
    exit_after_n_bars=10,  # Exit after 10 bars
)
engine = NautilusBacktestEngine(config)
# run() takes a boolean pd.Series aligned to the OHLCV DataFrame index
result = engine.run(pd.Series(signal.series, index=df.index), df)
metrics = engine.get_metrics(result)
```

The legacy backtesting.py adapter is still available when the `legacy` extra
is installed:

```python
from evoquant.backtest_engine.evo_bt import EvoStrategy

# Configure strategy parameters (class attributes)
EvoStrategy.direction = "LongOnly"  # or "ShortOnly", "LongShort"
EvoStrategy.trade_size = 0.99  # 99% of available capital
EvoStrategy.stop_loss = (2.0, 'Percent')  # 2% stop loss
EvoStrategy.exit_after_n_bars = 10  # Exit after 10 bars
```

### Fitness Functions

`PerfStats` takes `(name, weight)` pairs (positive weight = maximize,
negative = minimize). Available names include:

- **Sharpe** - Risk-adjusted returns
- **Sortino** - Downside deviation adjusted returns
- **Calmar** - Return relative to maximum drawdown
- **MaxDD** - Maximum peak-to-trough decline
- **CAGR/AvgDD**, **AvgDD**, **Volatility**, **VaR**, **CVaR**, **Stability**
- **Total$PnL**, **Avg$PnL**, **Avg$Profit**, **Avg$Loss**, **Max$Loss**
- **NumberOfTrades**, **MaxDD_Duration**

```python
from evoquant.evo_gp import PerfStats

# Define fitness functions with weights (variadic pairs, not a list)
perf_stats = PerfStats(
    ("Sharpe", 1.0),   # Maximize Sharpe ratio
    ("Sortino", 0.5),  # Maximize Sortino ratio
    ("MaxDD", -1.0),   # Minimize max drawdown
)
```

## Advanced Usage

### Code Translation

EvoQuant can render deployable trading code from Jinja2 templates:

```python
from evoquant.translator_engine.jinja_template import TemplateEngine

engine = TemplateEngine()  # loads templates from translator_engine/
code = engine.render_string("Hello {{ name }}!", name="World")
```

For full strategy translation, `CTraderTranslator` (built on a
`PTTranslator` primitive tree) exposes `register_primitive`,
`store_strat_config`, and `run_translation`; `CTraderJinjaTranslator` renders
the `ctrader_template.j2` template. See `tests/test_jinja_template.py` and
`tests/Example Generated Strategy/Generated Code for CSharp.cs`.

## Architecture

### Type System

EvoQuant uses a strongly-typed GP system:

- **ParameterBase** - Base class for parameters (`evoquant.base`)
  - **Period** - Lookback period (int >= 5)
  - **Lag** - Shift/lag value (int >= 1)
- Indicator parameters live in `evoquant.indicators`:
  - **StdDev** - Standard deviation multiplier (float > 0)
  - **MAMode** - Moving average mode ('ema', 'sma')
  - **BBOut** - Bollinger Bands output ('bbl', 'bbm', 'bbu', 'bbb', 'bbp')

### Backtesting Engine

NautilusTrader is the primary engine behind the `BacktestEngineAdapter`
interface (`run`, `get_metrics`, `validate_data`), with results standardized
as `BacktestResult`. The legacy `backtesting.py` (`evo_bt.py`) and `vectorbt`
paths remain available via the `legacy` extra. See `docs/ARCHITECTURE.md`.

## Testing

Run the test suite from the repo root (requires Python 3.12):

```bash
uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing
```

CI (`lint-and-format`, `test` jobs) additionally enforces:

```bash
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/
```

Test files with real test cases:
- `test_nautilus_adapter.py` - NautilusTrader engine tests
- `test_jinja_template.py` - Jinja2 template rendering and cTrader translation tests
- `test_backtesting.py` - Legacy backtesting tests (skipped unless vectorbt is installed)

Smoke/script files (imported at collection, define no test cases):
- `test_base.py`, `test_indicators.py`, `test_imports.py`, `test_string_code.py`, `test_utils.py`

## Examples

See `tests/Example Generated Strategy/` (`Generated Code for CSharp.cs`, `SP500.png`)
for example generated strategy output.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
