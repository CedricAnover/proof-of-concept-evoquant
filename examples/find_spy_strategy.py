"""End-to-end probe: evolve a trading strategy on daily SPY, emit cTrader C# code.

Pipeline exercised (all real project entry points, no mocks):
    yfinance OHLCV -> Series wrappers -> DEAP PrimitiveSetTyped (hand-wired;
    the repo ships no toolbox/pset factory) -> generate_composite_signal_root
    -> evo_evaluator (legacy evo_backtester + PerfStats + filter layer 1)
    -> best tree -> PTTranslator/CTraderTranslator -> cTrader C# file
    (+ Jinja2 render for comparison).

Requirements (examples-only, NOT project dependencies):
    yfinance  (already present in this env; otherwise: uv pip install yfinance)
    legacy extra installed (uv sync --all-extras --dev --extra legacy)
    for backtesting.py. vectorbt is NOT needed by this script.

Run from the repo root:
    uv run python examples/find_spy_strategy.py [--pop 16] [--start 2020-01-01]
                                                [--end 2025-01-01] [--no-download]

The script prints PROBE FINDINGS: every weakness in the implementation,
architecture, or design that this run exposes. Exit code 0 with a winner
means the full path worked; anything else is itself a finding.
"""

import argparse
import functools
import math
import os
import random
import sys

import numpy as np
import pandas as pd
from backtesting import Backtest
from deap import base, creator, gp, tools

from evoquant import (
    Period,
    SeriesBool,
    SeriesClose,
    SeriesFloat,
    SeriesHigh,
    SeriesIndicator,
    SeriesLow,
    SeriesOpen,
    SeriesVolume,
)
from evoquant.backtest_engine.evo_bt import EvoStrategy, evo_backtester, evo_filter_layer1
from evoquant.evo_gp import (
    PerfStats,
    eval_modifier,
    evo_compiler,
    evo_evaluator,
    generate_composite_signal_root,
)
from evoquant.indicators import RSI, rsi, sma
from evoquant.signals import (
    and_rule,
    cross_above_rule,
    cross_below_rule,
    is_above_rule,
    is_below_rule,
    not_rule,
    or_rule,
)
from evoquant.translator_engine.base import CTraderTranslator, PTTranslator
from evoquant.translator_engine.jinja_template import CTraderJinjaTranslator

FINDINGS: list[str] = []


def note(text: str) -> None:
    FINDINGS.append(text)
    print(f"  [PROBE] {text}")


note(
    "FINDING #1 (base.py SeriesBase) FIXED in-repo: int-dtype pd.Series input "
    "no longer crashes (int branch skips the NaN fill). This script now feeds "
    "raw yfinance int64 Volume with no workaround to exercise the fix."
)


# ---------------------------------------------------------------- data ---
def load_spy(start: str, end: str, allow_download: bool) -> pd.DataFrame:
    """Daily SPY OHLCV, cleaned for backtesting.py (tz-naive DatetimeIndex)."""
    try:
        import yfinance as yf
    except ImportError:
        note("yfinance is not installed (examples-only dep, correctly absent from pyproject).")
        raise SystemExit("Install it with: uv pip install yfinance") from None

    if not allow_download:
        note("--no-download given but no local cache exists; downloading anyway.")
    df = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise SystemExit("yfinance returned no data (network blocked? empty range?).")
    if isinstance(df.columns, pd.MultiIndex):  # single-ticker multi-level columns
        df.columns = df.columns.get_level_values(0)
    missing = {"Open", "High", "Low", "Close"} - set(df.columns)
    if missing:
        raise SystemExit(f"Missing OHLC columns after flatten: {missing}")
    cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
    df = df[cols].dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    note(f"SPY {df.index.min().date()} -> {df.index.max().date()}, {len(df)} daily bars.")
    return df


# ---------------------------------------------------------------- pset ---
PRIMITIVES = {  # closed world: everything GP may emit AND translation must cover
    "sma": (sma, [SeriesFloat, Period], SeriesFloat,
            "Indicators.SimpleMovingAverage({0}, {1}).Result"),
    "rsi": (rsi, [SeriesFloat, Period], SeriesFloat,
            "Indicators.RelativeStrengthIndex({0}, {1}).Result"),
    # cross_* have no exact cTrader equivalent; .Last(1) approximates prev-bar.
    "cross_above_rule": (cross_above_rule, [SeriesFloat, SeriesFloat], SeriesBool,
                         "({0} > {1} && {0}.Last(1) < {1}.Last(1))"),
    "cross_below_rule": (cross_below_rule, [SeriesFloat, SeriesFloat], SeriesBool,
                         "({0} < {1} && {0}.Last(1) > {1}.Last(1))"),
    "is_above_rule": (is_above_rule, [SeriesFloat, SeriesFloat], SeriesBool, "({0} > {1})"),
    "is_below_rule": (is_below_rule, [SeriesFloat, SeriesFloat], SeriesBool, "({0} < {1})"),
    "and_rule": (and_rule, [SeriesBool, SeriesBool], SeriesBool, "({0} && {1})"),
    "or_rule": (or_rule, [SeriesBool, SeriesBool], SeriesBool, "({0} || {1})"),
    "not_rule": (not_rule, [SeriesBool], SeriesBool, "(!{0})"),
}


def build_pset(wrapped: dict[str, SeriesFloat]) -> tuple[gp.PrimitiveSetTyped, dict]:
    """Hand-wired pset. The repo ships no factory: no PrimitiveSetTyped,
    no creator fitness, no toolbox registrations exist anywhere in-package."""
    note("No DEAP factory in-package: pset/creator/toolbox all hand-wired here "
         "(the Orchestrator/Evolver stubs were supposed to own this).")
    arg_names = ["open", "high", "low", "close", "volume"]
    pset = gp.PrimitiveSetTyped("MAIN", [SeriesFloat] * 5, SeriesBool)
    pset.renameArguments(**{f"ARG{i}": n for i, n in enumerate(arg_names)})
    for name, (func, arg_types, ret_type, _csharp) in PRIMITIVES.items():
        pset.addPrimitive(func, arg_types, ret_type, name=name)
    pset.addEphemeralConstant("period", lambda: Period(random.randint(5, 50)), Period)
    # Indicator track (the COMPILE_* pattern): sma/rsi return SeriesIndicator,
    # a type with NO terminals, so SeriesIndicator nodes can ONLY expand via an
    # indicator primitive. Without this, SeriesFloat nodes always collapse to raw
    # series terminals and indicators are unreachable (FINDING #5 below).
    note(
        "FINDING #5 (generation coverage): with SeriesFloat in terminal_types, "
        "SeriesFloat nodes always collapse to raw terminals, so sma/rsi primitives "
        "can never be generated (verified: 0 indicator nodes in first 16 trees). "
        "Reachable indicators require a separate no-terminal return type per the "
        "COMPILE_* pattern, which exists only as non-executable pseudo-code strings."
    )
    pset.addPrimitive(sma, [SeriesFloat, Period], SeriesIndicator, name="sma")
    pset.addPrimitive(rsi, [SeriesFloat, Period], SeriesIndicator, name="rsi")
    for cmp_name in ("cross_above_rule", "cross_below_rule", "is_above_rule", "is_below_rule"):
        func, _, _, _ = PRIMITIVES[cmp_name]
        pset.addPrimitive(func, [SeriesIndicator, SeriesIndicator], SeriesBool, name=cmp_name)
    # ARG terminals resolve as lambda params for evo_compiler AND as the
    # strings "open"/... for translate_terminal. Series-object terminals
    # would crash the translator (it only handles str/ParameterBase).
    note("Translator only accepts str/ParameterBase terminals: ARG-name "
         "terminals used; Series-object terminals would crash translate_terminal.")
    p_context = {name: func for name, (func, _, _, _) in PRIMITIVES.items()}
    p_context.update({"SeriesFloat": SeriesFloat, "SeriesBool": SeriesBool, "Period": Period})
    main_input = tuple(wrapped[n] for n in arg_names)
    return pset, p_context, main_input


def build_toolbox(pset, p_context, main_input, bt, evo_bt_params, perf_stats) -> base.Toolbox:
    if hasattr(creator, "FitnessMulti"):
        del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMulti", base.Fitness, weights=tuple(perf_stats.weights))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    toolbox.register(
        "expr", generate_composite_signal_root, pset, 1, 3, [SeriesFloat, Period]
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    evaluator = functools.partial(
        evo_evaluator,
        pset=pset,
        pset_mapping=p_context,
        main_input=main_input,
        bt=bt,
        evo_bt_params=evo_bt_params,
        perf_stats=perf_stats,
    )
    toolbox.register("evaluate", eval_modifier, evaluator=evaluator)
    return toolbox


# --------------------------------------------------------------- search ---
def evolve(toolbox, pop_size: int, perf_stats: PerfStats) -> list:
    pop = toolbox.population(n=pop_size)
    seen_errors: set[str] = set()
    for i, ind in enumerate(pop):
        try:
            toolbox.evaluate(ind)
        except Exception as err:  # noqa: BLE001 - probe must survive individuals
            key = f"{type(err).__name__}: {err}"
            if key not in seen_errors:  # dedupe: one note per distinct failure mode
                seen_errors.add(key)
                note(f"Individual {i} raised {key} "
                     f"(evo_evaluator re-raises non-Zero/Overflow errors).")
            ind.fitness.values = tuple(perf_stats.invalid_values)
        print(f"  ind {i}: {ind}  -> {tuple(ind.fitness.values)}")
    return pop


def build_diag_tree(pset) -> gp.PrimitiveTree:
    """Hand-build cross_above(sma(close,10), sma(close,20)) as a SeriesBool root.

    NOTE: is_above_rule(rsi, rsi) would be degenerate (x > x == False
    everywhere), and the pset has no constant-threshold primitive
    (series_*_value_rule exists in signals.py but is not wired here).
    """
    prims: dict[str, gp.Primitive] = {}
    for plist in pset.primitives.values():
        for p in plist:
            prims.setdefault(p.name, p)
    # NOTE: Terminal.__str__ is the default object repr; the usable key is .value
    # (PrimitiveTree.__str__ renders terminals via Terminal.format -> raw value).
    terms = {t.value: t for t in pset.terminals[SeriesFloat]}
    # Ephemeral nodes must be MetaEphemeral-wrapped (raw Period values have no
    # .arity and crash PrimitiveTree.__str__); instantiate the class DEAP
    # registered, then pin deterministic values.
    eph_cls = next(t for t in pset.terminals[Period] if isinstance(t, type))

    def period_node(v: int):
        node = eph_cls()
        node.value = Period(v)
        return node

    return gp.PrimitiveTree(
        [
            prims["cross_above_rule"],
            prims["sma"],
            terms["close"],
            period_node(10),
            prims["sma"],
            terms["close"],
            period_node(20),
        ]
    )


def diagnose(pset, p_context, main_input, bt, evo_bt_params, perf_stats) -> None:
    """Evaluate the hand-built diag tree with full internals exposed."""
    tree = build_diag_tree(pset)
    print(f"  tree: {tree}")
    func = evo_compiler(tree, pset, p_context)
    ser_bool = func(*main_input)
    print(f"  compiled -> {type(ser_bool).__name__}, "
          f"True bars: {int(np.asarray(ser_bool.series).sum())}/{ser_bool.size}")
    bt_res = evo_backtester(ser_bool, bt, **evo_bt_params)
    for split, (rets, trades) in bt_res.items():
        cum = (float((1 + rets).cumprod().iloc[-1]) if len(rets) else float("nan"))
        print(f"  {split}: {len(rets)} return bars, {trades.shape[0]} trades, "
              f"cumret={cum:.4f}, layer1={evo_filter_layer1(rets, trades)}")
    perf_stats.set_fitness_required_args(bt_res["IS"][0], bt_res["IS"][1])
    print(f"  PerfStats fitness_values={perf_stats.fitness_values} "
          f"(invalid={perf_stats.invalid_values})")


def pick_best(pop, perf_stats: PerfStats):
    def valid(ind) -> bool:
        vals = tuple(ind.fitness.values)
        return len(vals) == perf_stats.n_fitness and all(
            not (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
            for v in vals
        )

    candidates = [ind for ind in pop if valid(ind)]
    if not candidates:
        note("Every individual failed filter layer 1 (needs >=30 IS+OOS trades, "
             "positive drift, sane outliers): random search cannot clear it at this size.")
        return None
    best = max(candidates, key=lambda ind: float(ind.fitness.values[0]))
    note(f"{len(candidates)}/{len(pop)} individuals valid; best Sharpe={best.fitness.values[0]:.3f}.")
    return best


# ---------------------------------------------------------- translation ---
STRAT_PARAMS = {  # exactly the 12 _EVO_STRAT_PARAMS keys; mirrors backtest config
    "strategy_name": "SPYProbe",
    "direction": "LongOnly",
    "trade_size": 0.99,
    "exit_encoded_entry": True,
    "exit_after_n_bars": None,
    "exit_after_n_days": None,
    "exit_end_of_week": False,
    "exit_end_of_month": False,
    "exit_when_pnl_lessthan": None,
    "stop_loss": (),
    "take_profit": (),
    "enable_tsl": False,
}


def translate(best, outdir: str) -> None:
    tree = gp.PrimitiveTree(best) if not isinstance(best, gp.PrimitiveTree) else best
    used = {node.name for node in tree if isinstance(node, gp.Primitive)}
    unregistered = used - set(PRIMITIVES)
    if unregistered:
        note(f"Untranslatable primitives evolved: {unregistered} (closed-world registry).")
        return
    note("Closed-world translation holds only because the pset was restricted to 9 "
         "pre-registered primitives; shift/quantile/ma rules have no C# mapping.")

    for name, (_, _, _, csharp) in PRIMITIVES.items():
        CTraderTranslator.register_primitive(name, csharp)
    note("Primitive registry is CLASS-level mutable state: registrations leak "
         "across instances/runs (no isolation).")
    CTraderTranslator.store_strat_config(dict(STRAT_PARAMS))
    translator = CTraderTranslator(PTTranslator(tree))
    code = translator.get_complete_code(directory=outdir)
    legacy_path = os.path.join(outdir, "ctrader_legacy_note.txt")
    with open(legacy_path, "w") as f:
        f.write(f"root_signal: {translator.root_signal}\n")
    print(f"  legacy C# (string.Template) saved under {outdir}/ctrader_code_*.txt")

    jinja_params = dict(STRAT_PARAMS)
    jinja_params.update({"stop_loss": None, "take_profit": None})
    jinja = CTraderJinjaTranslator(
        indicator_signal_code=translator.indicator_signal_code, strategy_params=jinja_params
    )
    jinja_code = jinja.generate_code(root_signal=translator.root_signal, save_directory=outdir)
    print(f"  jinja C# saved under {outdir}/ (length {len(jinja_code)} chars)")
    for needle in ("namespace cAlgo.Robots", "RunEntries", translator.root_signal):
        if needle not in code or needle not in jinja_code:
            note(f"Generated code missing expected block: {needle!r}")


# ----------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="evaluate one hand-built tree with internals exposed, then exit")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(outdir, exist_ok=True)

    print("== 1. data ==")
    df = load_spy(args.start, args.end, allow_download=not args.no_download)
    wrapped = {
        "open": SeriesOpen(df["Open"]),
        "high": SeriesHigh(df["High"]),
        "low": SeriesLow(df["Low"]),
        "close": SeriesClose(df["Close"]),
        # PROBE FINDING #1 was fixed in-repo (int branch skips the NaN fill),
        # so raw int64 yfinance Volume is fed deliberately to exercise the fix.
        "volume": SeriesVolume(df["Volume"]) if "Volume" in df.columns else SeriesClose(df["Close"]),
    }

    print("== 2. DEAP wiring ==")
    pset, p_context, main_input = build_pset(wrapped)
    perf_stats = PerfStats(("Sharpe", 1.0), ("MaxDD", -1.0))
    bt = Backtest(df, EvoStrategy, cash=10_000, commission=0.0)
    # NOTE (PROBE FINDING #2, now fixed in-repo): evo_backtester() takes
    # **evo_bt_params where the strategy kwargs must hide one level deeper under
    # the "strat_params" key (forwarded to bt.run(**strat_params)). This is now
    # fail-fast validated (TypeError/ValueError) and documented in the docstring.
    # Passing direction/strategy_name at top level raises `TypeError:
    # evo_backtester() got an unexpected keyword argument 'direction'`. The first
    # full-population run died this way (16/16 invalid) before the fix.
    note(
        "FINDING #2 (evo_bt.py calling convention) FIXED in-repo: strategy kwargs "
        "must nest under evo_bt_params['strat_params']; violations now fail fast "
        "with TypeError/ValueError and the convention is docstring-documented."
    )
    evo_bt_params = {"strat_params": {"direction": "LongOnly", "strategy_name": "SPYProbe"}}
    if args.diag:
        print("== DIAG (single hand-built tree) ==")
        diagnose(pset, p_context, main_input, bt, evo_bt_params, perf_stats)
        print("== DIAG cTrader codegen (same tree) ==")
        translate(build_diag_tree(pset), outdir)
        return print_findings() or 0
    toolbox = build_toolbox(pset, p_context, main_input, bt, evo_bt_params, perf_stats)

    print("== 3. random search ==")
    pop = evolve(toolbox, args.pop, perf_stats)
    best = pick_best(pop, perf_stats)
    if best is None:
        print("No valid strategy. Findings above.")
        return print_findings() or 1

    print(f"== 4. winner ==\n  {best}\n  fitness={tuple(best.fitness.values)}")
    print("== 5. cTrader codegen ==")
    translate(best, outdir)
    return print_findings() or 0


def print_findings() -> int:
    print(f"\n== PROBE FINDINGS ({len(FINDINGS)}) ==")
    for i, f in enumerate(FINDINGS, 1):
        print(f"  {i}. {f}")
    return 0


if __name__ == "__main__":  # noqa: C901 - linear probe driver
    sys.exit(main())
