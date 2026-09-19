## Project Overview
EvoQuant evolves interpretable trading strategies with genetic programming (DEAP): typed indicators and boolean signal rules compose into strategy trees, which are fitness-scored by backtesting and translated into deployable cTrader C# code. It is for quantitative researchers and strategy developers exploring automated strategy discovery on OHLCV data. Current state is a research proof-of-concept mid-migration: the NautilusTrader adapter abstraction exists but the GP engine still drives the legacy backtesting.py path, and the Orchestrator/Evolver entry points are unimplemented stubs (see `docs/ARCHITECTURE.md`).

- **Architecture:** Modular monolith (Python package `src/evoquant/`): typed series/parameter system → indicators → signal rules → DEAP GP engine → backtest adapters → Jinja2/string code generation
- **Domain:** Finance — systematic trading strategy research (evolution, backtesting, cTrader code generation)
- **Scale:** Research proof-of-concept (no deployment target; single-machine evolution runs)

## Tech Stack
### Core Technologies
- **Language:** Python 3.12
- **Data:** yfinance, pandas, numpy, ta-lib-python, sklearn
- **Research:** deap, imodels, pysr
- **Backtest:** nautilustrader, pybroker, backtesting.py, backtrader, vectorbt
- **Performance:** quantstats
- **Package Manager:** uv

### Key Dependencies
- `deap` `>=1.3.0` — genetic programming engine (typed primitives, evolution loops)
- `pandas` `>=2.0.0` / `numpy` `>=1.24.0,<2.4.0` — OHLCV data handling and numeric core
- `pandas-ta` `>=0.3.14b` — technical indicator implementations (SMA, RSI, BBands, …)
- `nautilus-trader` `>=1.231.0` — primary backtesting/trading engine (adapter exists; engine integration in progress)
- `quantstats` `>=0.0.60` — single metrics backend for fitness scoring (Sharpe, drawdown, …)
- `scikit-learn` `>=1.0.0` / `statsmodels` `>=0.14.0` / `numba` `>=0.59.0` — train/test splits, stationarity stats, performance
- `jinja2` `>=3.1.0` — cTrader C# code-generation templates
- Legacy extra (`legacy`): `backtesting` `>=0.6.0`, `vectorbt` `>=1.0.0`, `plotly` `<7.0.0` (pinned: vectorbt 1.0.0 needs `scattermapbox`, removed in plotly 7.x)

### Development Tools
- **Linter:** Ruff (`uv run ruff check src/ tests/` — CI-gated, must pass)
- **Formatter:** Ruff format (`uv run ruff format --check src/ tests/` — CI-gated, must pass)
- **Type Checker:** mypy (CI runs it non-blocking via `|| true`)
- **Testing:** pytest + pytest-cov (`uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing`)

## Setup Commands
```bash
# Install dependencies
uv sync --all-extras --dev

# Add legacy backtesting engines (backtesting.py, vectorbt)
uv sync --all-extras --dev --extra legacy
```

## Build, Test, and Lint Commands

### File-Scoped Commands (Preferred — Fast)
Use these when checking individual changes. They complete in seconds.
```bash
# Type-check a single file
uv run mypy path/to/file.ext --ignore-missing-imports

# Lint a single file
uv run ruff check path/to/file.ext

# Run a single test file
uv run pytest tests/test_file.py -v --tb=short
```

### Project-Wide Commands (Use Sparingly)
Run these only before a final commit or when explicitly requested.
```bash
# Full type check
uv run mypy src/evoquant/ --ignore-missing-imports --no-strict-optional || true

# Full test suite
uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing

# Full build
# (no build step; package is pure Python)

# Full lint
uv run ruff format --check src/ tests/
uv run ruff check src/ tests/
```

> **Rule:** Always use file-scoped commands for iterative development. Only run project-wide commands before creating a pull request.

## Project Structure
```
/
├── .github/workflows/ci-cd.yml   # CI: lint-and-format, test, integration, docs
├── docs/ARCHITECTURE.md          # Architecture documentation
├── examples/find_spy_strategy.py # End-to-end SPY→cTrader probe script
├── pyproject.toml                # uv project config (deps, ruff, mypy, pytest)
├── uv.lock                       # Pinned dependency lockfile
├── src/evoquant/                 # Core package
│   ├── __init__.py               # Public API exports
│   ├── base.py                   # Series/Parameter type system
│   ├── evo_gp.py                # DEAP GP engine (still legacy-coupled)
│   ├── indicators.py            # Technical indicators
│   ├── signals.py               # Boolean signal rules
│   ├── orchestrator.py          # Orchestrator/Evolver (stubs)
│   ├── backtest_engine/         # Backtesting adapters
│   │   ├── __init__.py
│   │   ├── base.py              # BacktestConfig, BacktestEngineAdapter, BacktestResult, Trade, MetricsCalculator, BacktestError
│   │   ├── nautilus_adapter.py  # NautilusBacktestEngine (primary, pandas simulation)
│   │   ├── evo_bt.py            # Legacy backtesting.py adapter
│   │   ├── utils.py             # Backtesting utilities
│   │   └── validation.py        # Train/test and walk-forward split logic
│   └── translator_engine/       # Code translation engines
│       ├── __init__.py
│       ├── base.py              # PTTranslator, CTraderTranslator (string.Template)
│       ├── jinja_template.py    # TemplateEngine, CTraderJinjaTranslator (Jinja2)
│       ├── ctrader_template.j2  # cTrader Jinja2 template
│       └── ctrader_template.cs  # Example generated C# output
└── tests/                        # Test suite
    ├── __init__.py
    ├── test_nautilus_adapter.py
    ├── test_jinja_template.py
    ├── test_backtesting.py
    ├── test_base.py
    ├── test_indicators.py
    ├── test_imports.py
    ├── test_string_code.py
    ├── test_utils.py
    └── Example Generated Strategy/
        ├── Generated Code for CSharp.cs
        └── SP500.png
```

### Key Files
- `pyproject.toml` — Dependency versions, ruff config (line-length 120, select E/F/W/I/N/UP/B/C4/SIM, ignore E501), mypy settings, pytest config
- `src/evoquant/base.py` — Series/Parameter type system (SeriesFloat, SeriesBool, Period, Lag); int-dtype crash fixed
- `src/evoquant/evo_gp.py` — DEAP GP engine (PerfStats 17 metrics, evo_evaluator, gp_main_algo_random/standard_gp); still calls legacy evo_backtester directly
- `src/evoquant/backtest_engine/base.py` — New abstraction (BacktestEngineAdapter run/get_metrics/validate_data, BacktestConfig, BacktestResult, Trade, MetricsCalculator, BacktestError)
- `src/evoquant/backtest_engine/nautilus_adapter.py` — NautilusBacktestEngine (pandas simulation, no nautilus_trader import)
- `src/evoquant/backtest_engine/evo_bt.py` — Legacy backtesting.py adapter (EvoStrategy, evo_backtester, filter layers)
- `src/evoquant/translator_engine/base.py` — PTTranslator/CTraderTranslator (string.Template C# codegen)
- `src/evoquant/translator_engine/jinja_template.py` — TemplateEngine/CTraderJinjaTranslator (Jinja2 C# codegen)
- `tests/test_nautilus_adapter.py` — Only test file with real test cases (2 tests)
- `tests/test_backtesting.py` — Legacy backtesting tests (1 test, vectorbt-gated skip)
- `examples/find_spy_strategy.py` — End-to-end SPY→cTrader probe script (finds weaknesses)

## Code Style and Conventions

### Language-Specific Rules
- All Python functions require type hints
- Prefer composition over inheritance
- Use explicit dependencies (dependency injection)
- Validate at boundaries

### Naming Conventions
- **Files:** lowercase-with-dashes.py
- **Variables:** descriptive (userCount not uc), const by default
- **Functions:** verbPhrases (getUser, validateEmail)
- **Classes:** PascalCase, noun-based names
- **Constants:** UPPER_SNAKE_CASE

### Formatting Rules
- **Indentation:** 4 spaces
- **Line length:** 120 characters max
- **Quotes:** single quotes
- **Semicolons:** no
- **Trailing commas:** yes

### Do's ✅
- Pure functions (same input = same output, no side effects)
- Immutability (create new data, don't modify)
- Composition (build complex from simple)
- Small functions (< 50 lines)
- Explicit dependencies (dependency injection)

### Don'ts ❌
- Mutation, side effects, deep nesting
- God modules, global state, large functions

### Code Example
```python
# ✅ Pure function
from evoquant.base import SeriesFloat

def compute_sma(series: SeriesFloat, period: int) -> SeriesFloat:
    """Compute simple moving average."""
    return sma(series, Period(period))

# ❌ Impure (side effects)
def update_strategy(state):
    state["count"] += 1  # mutation
    return state
```

## Testing Instructions

### Framework and Location
- **Unit tests:** pytest — `tests/test_*.py` alongside source files
- **Integration tests:** pytest with `@pytest.mark.integration` — `tests/`
- **E2E tests:** N/A (research proof-of-concept)

### Running Tests
```bash
# Run all tests
uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_nautilus_adapter.py -v --tb=short

# Run with coverage
uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=xml --cov-report=term-missing

# Run in watch mode
uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing -f
```

### Test Requirements
- **Coverage target:** no gate (CI runs coverage non-blocking)
- **Test naming:** `test_*` functions, `Test*` classes
- **Mocking strategy:** use real data where possible; mock external APIs

### Testing Patterns
```python
# Well-structured test example from tests/test_nautilus_adapter.py
class TestNautilusBacktestEngine:
    """Test cases for NautilusBacktestEngine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Open": np.random.randn(100).cumsum() + 100,
                "High": np.random.randn(100).cumsum() + 101,
                "Low": np.random.randn(100).cumsum() + 99,
                "Close": np.random.randn(100).cumsum() + 100,
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )
        return data

    def test_engine_initialization(self):
        """Test that the engine can be initialized with default config."""
        engine = NautilusBacktestEngine()
        assert engine is not None
        assert engine.config.initial_cash == 100000.0
```

## Git Workflow

### Branch Naming
- **Features:** `feature/[ticket-id]-brief-description`
- **Bug fixes:** `fix/[ticket-id]-brief-description`
- **Hotfixes:** `hotfix/brief-description`

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
type(scope): description
```
**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
- `feat(auth): add password reset flow`
- `fix(api): handle null from external service`

### Pull Request Requirements
Before creating a PR, all of the following must pass:
- [ ] `uv run ruff format --check src/ tests/` — zero errors
- [ ] `uv run ruff check src/ tests/` — zero errors
- [ ] `uv run pytest tests/ -v --tb=short --cov=src/evoquant --cov-report=term-missing` — all tests green
- [ ] Documentation updated if behavior changed
- [ ] Debug logs and commented-out code removed

**PR Title:** `[type(scope): description]`
**PR Description must include:** Summary, related issue number, testing performed, breaking changes (if any).

## Boundaries and Safety

### ✅ Allowed Without Asking
- Read files and list directory contents
- Type-check, lint, and format a single file
- Run a single unit test
- Search the codebase and read documentation
- Create feature branches and commits with descriptive messages

### ⚠️ Require Approval First
- Installing or updating dependencies (`uv sync`)
- Modifying configuration files (`pyproject.toml`, `.github/workflows/`, `uv.lock`)
- Running the full project build or full test suite
- Git push operations
- Deleting files or directories
- Modifying database schemas
- Changing environment variables

### ✗ Never Do
- **Never commit secrets, API keys, or credentials.**
- **Never push directly to `main` or `develop`** — work on feature branches
- **Never force-push shared branches**
- **Never hardcode secrets, keys, or passwords** — load from environment or ignored config

## Security Considerations

### Authentication & Authorization
- This is a research proof-of-concept — no web API or user authentication in-repo
- Trading platform authentication (cTrader, NautilusTrader) is handled at the platform level, not in this codebase
- Backtest engines operate on local OHLCV data; no multi-tenant access control

### Input Validation
- Validate all external data at boundaries: OHLCV DataFrames, Series inputs, parameter values
- The typed Series system (`SeriesFloat`, `SeriesBool`, `Period`, `Lag`) enforces dtype and range constraints at construction time (e.g., `Period >= 5`, `Lag >= 1`, `StdDev > 0`)
- `BacktestEngineAdapter.validate_data()` requires Open/High/Low/Close columns, DatetimeIndex, non-empty data
- Sanitize file paths in examples and scripts (no path traversal)

### Secrets Management
- **Never commit secrets.** Use environment variables
- If adding API keys (yfinance, broker APIs) in the future, load from environment and add `.env` to `.gitignore`

### Security Requirements
- No database queries — data is in-memory pandas DataFrames and local CSV files
- Validate all external input; do not trust user or network data
- Apply least-privilege access; minimize exposed attack surface
- Keep dependencies pinned (`uv.lock`) and audit regularly

## Definition of Done
A task is complete only when **all** of the following are true:
- [ ] Code implements the requested behavior
- [ ] New or modified code has appropriate test coverage
- [ ] All existing tests still pass
- [ ] `uv run ruff format --check src/ tests/` — zero errors
- [ ] `uv run ruff check src/ tests/` — zero errors
- [ ] Documentation is updated (if applicable)
- [ ] No debug logs, `console.log`, or commented-out code remains

## Deployment

### CI/CD Pipeline
- **CI runs on:** GitHub Actions — see `.github/workflows/`

## When Stuck or Uncertain
- Ask clarifying questions rather than making assumptions.
- Propose a plan before implementing complex changes.
- Reference existing patterns in the codebase for consistency.
- Start small, implement a minimal solution first, then iterate.
- Write tests first when fixing bugs or adding features.
- Request review before making significant architectural changes.
