# Architecture Decision Record: Backtesting Engine Abstraction

## Context
The EvoQuant project currently uses `backtesting.py` and `vectorbt` for backtesting trading strategies. 
We need to replace these with NautilusTrader while maintaining the ability to swap backtesting engines 
in the future without affecting the rest of the codebase.

## Decision
Create an abstract backtesting engine interface that:
1. Defines a clear contract for any backtesting implementation
2. Allows easy swapping of backtesting engines (NautilusTrader, backtesting.py, vectorbt, etc.)
3. Separates concerns between strategy definition, execution, and performance measurement
4. Follows SOLID principles, particularly Dependency Inversion

## Architecture

### Module Structure
```
evoquant/
├── backtest_engine/
│   ├── __init__.py           # Public API exports
│   ├── base.py               # Abstract base classes & interfaces (NEW)
│   ├── nautilus_adapter.py   # NautilusTrader implementation (NEW)
│   ├── evo_bt.py             # Legacy backtesting.py adapter (to be refactored)
│   ├── utils.py              # Utility functions
│   └── validation.py         # Validation logic
```

### Key Abstractions

#### 1. BacktestEngineAdapter (Abstract Base Class)
- Defines the interface all backtesting engines must implement
- Methods: `run()`, `get_results()`, `get_metrics()`
- Properties: configuration, data requirements

#### 2. StrategyDefinition
- Platform-agnostic strategy representation
- Contains: entry signals, exit signals, parameters
- Can be compiled to different backtesting engine formats

#### 3. BacktestResult
- Standardized result object regardless of engine used
- Contains: trades, equity curve, metrics
- Provides consistent API for fitness evaluation

#### 4. MetricsCalculator
- Pluggable metrics calculation
- Supports Sharpe, Sortino, Calmar, Max Drawdown, etc.
- Engine-agnostic

### Dependencies

#### Required
- `nautilus-trader` - Primary backtesting engine
- `deap` - Genetic programming
- `pandas`, `numpy` - Data manipulation
- `pandas-ta` - Technical indicators

#### Optional/Dev
- `pytest`, `pytest-cov` - Testing
- `ruff` - Linting
- `mypy` - Type checking

### Migration Plan

1. **Phase 1**: Create abstract base classes and interfaces
2. **Phase 2**: Implement NautilusTrader adapter
3. **Phase 3**: Create wrapper for existing backtesting.py code
4. **Phase 4**: Update GP engine to use new abstraction
5. **Phase 5**: Remove vectorbt dependency
6. **Phase 6**: Clean up and finalize

## Consequences

### Positive
- ✅ Easy to swap backtesting engines
- ✅ Better separation of concerns
- ✅ More testable code
- ✅ Clear contracts between components
- ✅ Future-proof architecture

### Negative
- ⚠️ Initial development overhead
- ⚠️ Learning curve for NautilusTrader
- ⚠️ Need to maintain compatibility layer during migration

### Risks
- NautilusTrader may not support all features from backtesting.py
- Performance characteristics may differ
- Need to validate results match between engines

## References
- [NautilusTrader Documentation](https://nautilustrader.io/)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
