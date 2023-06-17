import copy

from deap import gp
import textwrap

import importlib
import inspect

from typing import Union, List

def store_imported_items(module_name, pkg_name):
    # Import the module dynamically
    module = importlib.import_module(module_name, package=pkg_name)
    # Get the current global symbol table
    global_symbols = {}
    # for k,v in globals().items():
    #     global_symbols[k] = v

    # Iterate over the attributes of the module and add them to globals()
    for name, item in inspect.getmembers(module):
        if (inspect.isclass(item) or inspect.isfunction(item)) and not name.startswith('__'):
            global_symbols[name] = item
    # print(global_symbols)
# store_imported_items('..base', 'evoquant.base')
# store_imported_items('..indicators', 'evoquant.indicators')
# store_imported_items('..signals', 'evoquant.signals')
store_imported_items('..test_main', 'evoquant.tests.test_main')

import os
# Function to clear the console log
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

filepath = r"D:\Projects\FinDashAnalytics\Data\evoquant\test_strategies.pickle"
import pickle
with open(filepath, "rb") as file:
    data = pickle.load(file)
population = data["population"] # List[gp.PrimitiveTree]
strategy_config = data["strat_config"] # List[gp.PrimitiveTree]

print(len(strategy_config))
for k,v in strategy_config.items():
    print(k, "=", v)

from evoquant.base import ParameterBase, SeriesBase, SeriesBool

from evoquant.translator_engine.base import CTraderTranslator, PTTranslator

# Register the Primitives in CTraderTranslator Class (not instance). These will be included as part of the main program. These are the template for mapping deap primitives to ctrader.
CTraderTranslator.register_primitive("shift", r"""Indicators.GetIndicator<Shift>({0}, {1}).Result""") # (<Series>, <Lag>)
CTraderTranslator.register_primitive("sum_indicator", r"""Indicators.GetIndicator<Sum>({0}, {1}).Result""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("diff", r"""Indicators.GetIndicator<Diff>({0}, {1}).Result""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("abs_diff", r"""Indicators.GetIndicator<AbsDiff>({0}, {1}).Result""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("abs_value", r"""Indicators.GetIndicator<AbsValue>({0}).Result""") # (<Series>)
CTraderTranslator.register_primitive("sma", r"""Indicators.SimpleMovingAverage({0}, {1}).Result""") # (<Series>, <Period>)
CTraderTranslator.register_primitive("highest", r"""Indicators.GetIndicator<Highest>({0}, {1}).Result""") # (<Series>)
CTraderTranslator.register_primitive("lowest", r"""Indicators.GetIndicator<Lowest>({0}, {1}).Result""") # (<Series>)
CTraderTranslator.register_primitive("rsi", r"""Indicators.RelativeStrengthIndex({0}, {1}).Result""") # (<Series>, <Period>)
CTraderTranslator.register_primitive("bbands", r"""Indicators.GetIndicator<BBands>({0}, {1}, {2}, {3}).{4}""") # (<Series>, <Period>, <Std>, <MAMode>, <BBOut>)
CTraderTranslator.register_primitive("zscore", r"""Indicators.GetIndicator<ZScore>({0}, {1}).Result""") # (<Series>, <Period>)

CTraderTranslator.register_primitive("and_rule", r"""{0} & {1}""") # (<Signal1>, <Signal2>)
CTraderTranslator.register_primitive("and_rule3", r"""{0} & {1} & {2}""") # (<Signal1>, <Signal2>, <Signal3>)
CTraderTranslator.register_primitive("and_rule4", r"""{0} & {1} & {2} & {3}""") # (<Signal1>, <Signal2>, <Signal3>, <Signal4>)
CTraderTranslator.register_primitive("or_rule", r"""{0} | {1}""") # (<Signal1>, <Signal2>)
CTraderTranslator.register_primitive("or_rule3", r"""{0} | {1} | {2}""") # (<Signal1>, <Signal2>, <Signal3>)
CTraderTranslator.register_primitive("or_rule4", r"""{0} | {1} | {2} | {3}""") # (<Signal1>, <Signal2>, <Signal3>, <Signal4>)
CTraderTranslator.register_primitive("xor_rule", r"""{0} ^ {1}""") # (<Signal1>, <Signal2>)
CTraderTranslator.register_primitive("xor_rule3", r"""{0} ^ {1} ^ {2}""") # (<Signal1>, <Signal2>, <Signal3>)
CTraderTranslator.register_primitive("xor_rule4", r"""{0} ^ {1} ^ {2} ^ {3}""") # (<Signal1>, <Signal2>, <Signal3>, <Signal4>)
CTraderTranslator.register_primitive("not_rule", r"""!{0}""") # (<Signal>)

CTraderTranslator.register_primitive("and2_or1", r"""({0} & {1}) | ({2} & {3})""")
CTraderTranslator.register_primitive("and3_or1", r"""({0} & {1} & {2}) | ({3} & {4} & {5})""")
CTraderTranslator.register_primitive("and2_or2", r"""({0} & {1}) | ({2} & {3}) | ({4} & {5})""")
CTraderTranslator.register_primitive("and3_or2", r"""({0} & {1} & {2}) | ({3} & {4} & {5}) | ({6} & {7} & {8})""")
CTraderTranslator.register_primitive("and2_or3", r"""({0} & {1}) | ({2} & {3}) | ({4} & {5}) | ({6} & {7})""")
CTraderTranslator.register_primitive("and3_or3", r"""({0} & {1} & {2}) | ({3} & {4} & {5}) | ({6} & {7} & {8}) | ({9} & {10} & {11})""")

CTraderTranslator.register_primitive("or2_and1", r"""({0} | {1}) & ({2} | {3})""")
CTraderTranslator.register_primitive("or3_and1", r"""({0} | {1} | {2}) & ({3} | {4} | {5})""")
CTraderTranslator.register_primitive("or4_and1", r"""({0} | {1} | {2} | {3}) & ({4} | {5} | {6} | {7})""")

CTraderTranslator.register_primitive("or2_and2", r"""({0} | {1}) & ({2} | {3}) & ({4} | {5})""")
CTraderTranslator.register_primitive("or3_and2", r"""({0} | {1} | {2}) & ({3} | {4} | {5}) & ({6} | {7} | {8})""")
CTraderTranslator.register_primitive("or4_and2", r"""({0} | {1} | {2} | {3}) & ({4} | {5} | {6} | {7}) & ({8} | {9} | {10} | {11})""")

CTraderTranslator.register_primitive("or2_and3", r"""({0} | {1}) & ({2} | {3}) & ({4} | {5}) & ({6} | {7})""")
CTraderTranslator.register_primitive("or3_and3", r"""({0} | {1} | {2}) & ({3} | {4} | {5}) & ({6} | {7} | {8}) & ({9} | {10} | {11})""")
CTraderTranslator.register_primitive("or4_and3", r"""({0} | {1} | {2} | {3}) & ({4} | {5} | {6} | {7}) & ({8} | {9} | {10} | {11}) & ({12} | {13} | {14} | {15})""")

CTraderTranslator.register_primitive("cross_above_rule", r"""SeriesCrossAboveSeries({0}, {1})""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("cross_below_rule", r"""SeriesCrossBelowSeries({0}, {1})""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("is_above_rule", r"""SeriesIsAboveSeries({0}, {1})""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("is_below_rule", r"""SeriesIsBelowSeries({0}, {1})""") # (<Series1>, <Series2>)
CTraderTranslator.register_primitive("is_incr_n_bars_rule", r"""IsIncr({0}, {1})""") # (<Series>, <IsIncrDecrNBars>)
CTraderTranslator.register_primitive("is_decr_n_bars_rule", r"""IsDecr({0}, {1})""") # (<Series>, <IsIncrDecrNBars>)
CTraderTranslator.register_primitive("is_highest_n_bars_rule", r"""IsHighest({0}, {1})""") # (<Series>, <HighLowNBars>)
CTraderTranslator.register_primitive("is_lowest_n_bars_rule", r"""IsLowest({0}, {1})""") # (<Series>, <HighLowNBars>)

CTraderTranslator.register_primitive("day_of_week_rule", r"""{0}.Last(1+0).DayOfWeek == {1}""") # (<SeriesData>, <DayOfWeek.value>)
CTraderTranslator.register_primitive("month_in_year_rule", r"""{0}.Last(1+0).Month == {1}""") # (<SeriesData>, <MonthInYear.value>)
CTraderTranslator.register_primitive("hour_in_day_rule", r"""{0}.Last(1+0).Hour ==  {1}""")
CTraderTranslator.register_primitive("hour_in_day_ge_rule", r"""{0}.Last(1+0).Hour >=  {1}""")
CTraderTranslator.register_primitive("hour_in_day_le_rule", r"""{0}.Last(1+0).Hour <=  {1}""")

CTraderTranslator.register_primitive("series_above_quantile_rule", r"""SeriesIsAboveSeries({0}, Indicators.GetIndicator<Quantile>({0}, {1}, {2}).Result)""") # (<Series>, <Quantile>, <Period>)
CTraderTranslator.register_primitive("series_below_quantile_rule", r"""SeriesIsBelowSeries({0}, Indicators.GetIndicator<Quantile>({0}, {1}, {2}).Result)""") # (<Series>, <Quantile>, <Period>)
CTraderTranslator.register_primitive("series_cross_above_quantile_rule", r"""SeriesCrossAboveSeries({0}, Indicators.GetIndicator<Quantile>({0}, {1}, {2}).Result)""") # (<Series>, <Quantile>, <Period>)
CTraderTranslator.register_primitive("series_cross_below_quantile_rule", r"""SeriesCrossBelowSeries({0}, Indicators.GetIndicator<Quantile>({0}, {1}, {2}).Result)""") # (<Series>, <Quantile>, <Period>)

CTraderTranslator.register_primitive("series_above_ma_rule", r"""SeriesIsAboveSeries({0}, Indicators.SimpleMovingAverage({0}, {1}).Result)""") # (<Series>, <Period>)
CTraderTranslator.register_primitive("series_below_ma_rule", r"""SeriesIsBelowSeries({0}, Indicators.SimpleMovingAverage({0}, {1}).Result)""") # (<Series>, <Period>)
CTraderTranslator.register_primitive("series_cross_above_ma_rule", r"""SeriesCrossAboveSeries({0}, Indicators.SimpleMovingAverage({0}, {1}).Result)""") # (<Series>, <Period>)
CTraderTranslator.register_primitive("series_cross_below_ma_rule", r"""SeriesCrossBelowSeries({0}, Indicators.SimpleMovingAverage({0}, {1}).Result)""") # (<Series>, <Period>)

CTraderTranslator.register_primitive("series_above_shift_rule", r"""SeriesIsAboveSeries({0}, Indicators.GetIndicator<Shift>({0}, {1}).Result)""") # (<Series>, <Lag>)
CTraderTranslator.register_primitive("series_below_shift_rule", r"""SeriesIsBelowSeries({0}, Indicators.GetIndicator<Shift>({0}, {1}).Result)""") # (<Series>, <Lag>)
CTraderTranslator.register_primitive("series_cross_above_shift_rule", r"""SeriesCrossAboveSeries({0}, Indicators.GetIndicator<Shift>({0}, {1}).Result)""") # (<Series>, <Lag>)
CTraderTranslator.register_primitive("series_cross_below_shift_rule", r"""SeriesCrossBelowSeries({0}, Indicators.GetIndicator<Shift>({0}, {1}).Result)""") # (<Series>, <Lag>)

CTraderTranslator.register_primitive("series_above_value_rule", r"""{0}.Last(1) > {1}""") # (<Series>, <ParameterBase.value>)
CTraderTranslator.register_primitive("series_below_value_rule", r"""{0}.Last(1) < {1}""") # (<Series>, <ParameterBase.value>)
CTraderTranslator.register_primitive("series_cross_above_value_rule", r"""{0}.Last(1) > {1} & {0}.Last(1+1) < {1}""") # (<Series>, <ParameterBase.value>)
CTraderTranslator.register_primitive("series_cross_below_value_rule", r"""{0}.Last(1) < {1} & {0}.Last(1+1) > {1}""") # (<Series>, <ParameterBase.value>)

# Store EvoStrategy Configurations
CTraderTranslator.store_strat_config(strat_params=strategy_config)


for INDIVIDUAL_IDX in range(len(population[:])):
    print("\n")
    print(fr"=================================================================== Testing Individual {INDIVIDUAL_IDX} ===================================================================")
    # INDIVIDUAL_IDX = 1 #Good Examples: 506, 6
    prim_tree = population[INDIVIDUAL_IDX]

    ptt = PTTranslator(prim_tree)

    # Skip iteration if there is no Composite Signal. We only want to investigate strategies with Composite Signal
    # if len(ptt.composite_signal_primitive_ls) == 0:
    #     continue

    print(prim_tree, "\n")
    print("Tree Length:", ptt.length)
    print("Tree Height:", ptt.height)
    print("Primitives", ptt.primitive_ls)
    print("Signals", ptt.signal_primitive_ls)
    print("Indicators", ptt.indicator_primitive_ls)
    print("Composite Signals", ptt.composite_signal_primitive_ls)
    print("Simple Signals", ptt.simple_signal_primitive_ls)
    print("Composite Indicators", ptt.composite_indicator_primitive_ls)
    print("Simple Indicators", ptt.simple_indicator_primitive_ls)
    print("Terminals", ptt.non_primitive_ls)
    print()
    for i in range(len(prim_tree)):
        print(f"Subtrees of {i}:", ptt.get_idx_subtrees(i))
    for i in range(len(prim_tree)):
        print(f"Args of {i}:", ptt.get_idx_args(i))
    for i in range(len(prim_tree)):
        print(f"Children & Grandchildren of {i}:", ptt.get_idx_child_grandchild(i))
    for i in range(len(prim_tree)):
        print(f"The Parent of {i} is ", ptt.get_idx_parent(i))
    print()

    ct_translator = CTraderTranslator(ptt) # Inputs must be a valid CTRADER syntax.

    # print("Testing run_translation", "\n")
    # ct_translator.run_translation()

    # print("Testing generate_indicator_signal_code", "\n")
    # ct_translator.generate_indicator_signal_code()

    assert isinstance(ct_translator.root_signal, str), "root_signal must be a string!"
    assert len(ct_translator.root_signal) != 0, f"Length of Root Signal for Individual={INDIVIDUAL_IDX} cannot be zero! Fix the problem!"
    assert ct_translator.root_signal != "", f"Root Signal for Individual={INDIVIDUAL_IDX} not modified! Fix the problem!"

    print(ct_translator.indicator_signal_code)
    # print()
    # print("Root Signal Full Expression", "\n", ct_translator.root_signal)

    # Generate Complete Valid cTrader C# Code
    ct_translator.get_complete_code(directory=r"D:\Projects\FinDashAnalytics\Data\evoquant\Strategy cTrader Codes")
    # ct_translator.get_complete_code(directory=None)

    print(r"#"*200)
    clear_console()  # Clear the console log
