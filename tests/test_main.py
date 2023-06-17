import concurrent.futures
import multiprocessing

import empyrical

from evoquant.base import *
from evoquant.indicators import *
from evoquant.signals import *
from evoquant.evo_gp import generate_safe, evo_compiler, generate_2permutations, generate_3permutations, evo_mutation, evo_cross, is_fitness_tuple_valid
from evoquant.backtest_engine.evo_bt import EvoStrategy, evo_backtester, evo_filter_layer1, evo_filter_layer2

import pandas as pd
import numpy as np

from backtesting import Backtest

from deap import gp, base, creator, tools
import copy
import random
import itertools

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)  # Ignore runtime warnings. This happens when there are some errors in calculating performance statistics.


"""
Solved my issue with https://github.com/DEAP/deap/issues/237 for generating random valid tree.
With this, I can specify all the terminal types.
"""

df_ohlcv = pd.read_csv(r"D:\Projects\FinDashAnalytics\Data\CTraderData\Clean\export-US500-Daily-BarChartHist.csv")
df_ohlcv['Volume'] = df_ohlcv['Volume'].astype(float) # Convert Volume to Float and not int
# Date here is a string if not transformed
# df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date']).dt.date
# df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])

# Specify all possible Terminal Types. These will be generated automatically, see example in RSI Code template.
terminal_types = []
terminal_types = [Period, Lag,
                  StdDev, MAMode, BBOut,
                  IncrDecrNBars, HighLowNBars, HourInDay, DayOfWeek, MonthInYear, Quantile,
                  SeriesDate, SeriesOpen, SeriesHigh, SeriesLow, SeriesClose, SeriesVolume,
                  SeriesPrice
                  ]

input_gp = (SeriesDate(df_ohlcv['Date']),
            SeriesPrice(df_ohlcv['Open']),
            SeriesPrice(df_ohlcv['High']),
            SeriesPrice(df_ohlcv['Low']),
            SeriesPrice(df_ohlcv['Close']),
            SeriesVolume(df_ohlcv['Volume']),
            SeriesOpen(df_ohlcv['Open']),
            SeriesHigh(df_ohlcv['High']),
            SeriesLow(df_ohlcv['Low']),
            SeriesClose(df_ohlcv['Close'])
            )

pset = gp.PrimitiveSetTyped("main",
                            [SeriesDate, SeriesPrice, SeriesPrice, SeriesPrice, SeriesPrice, SeriesVolume,
                             SeriesOpen, SeriesHigh, SeriesLow, SeriesClose],
                            SeriesBool
                            )

pset.renameArguments(ARG0="Date")
pset.renameArguments(ARG1="Open")
pset.renameArguments(ARG2="High")
pset.renameArguments(ARG3="Low")
pset.renameArguments(ARG4="Close")
pset.renameArguments(ARG5="Volume")
pset.renameArguments(ARG6="Open_")
pset.renameArguments(ARG7="High_")
pset.renameArguments(ARG8="Low_")
pset.renameArguments(ARG9="Close_")

#=======================================================================================================================

pset.addEphemeralConstant("Period", lambda: Period(random.randint(5, 200)), Period)

pset.addEphemeralConstant("Lag", lambda: Lag(random.randint(1, 30)), Lag) # Generate Random Instance of Lag(ParameterBase)
for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(shift, [combo, Lag], SeriesFloat, name="shift")
    pset.addPrimitive(shift, [combo, Lag], SeriesIndicator, name="shift")
pset.addPrimitive(shift, [SeriesPrice, Lag], SeriesPrice, name="shift")
pset.addPrimitive(shift, [SeriesVolume, Lag], SeriesVolume, name="shift")

for combo in generate_2permutations([SeriesFloat, SeriesIndicator, SeriesPrice, SeriesVolume]):
    pset.addPrimitive(sum_indicator, [combo[0], combo[1]], SeriesIndicator, name="sum_indicator")
    pset.addPrimitive(diff, [combo[0], combo[1]], SeriesIndicator, name="diff")
    pset.addPrimitive(abs_diff, [combo[0], combo[1]], SeriesIndicator, name="abs_diff")
    #---
    pset.addPrimitive(sum_indicator, [combo[0], combo[1]], SeriesFloat, name="sum_indicator")
    pset.addPrimitive(diff, [combo[0], combo[1]], SeriesFloat, name="diff")
    pset.addPrimitive(abs_diff, [combo[0], combo[1]], SeriesFloat, name="abs_diff")

for combo in [SeriesFloat, SeriesIndicator]:
    pset.addPrimitive(abs_value, [combo], SeriesIndicator, name="abs_value")
    pset.addPrimitive(abs_value, [combo], SeriesFloat, name="abs_value")

for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(sma, [combo, Period], SeriesIndicator, name="sma")
    pset.addPrimitive(sma, [combo, Period], SeriesFloat, name="sma")
pset.addPrimitive(sma, [SeriesPrice, Period], SeriesPrice, name="sma")
pset.addPrimitive(sma, [SeriesVolume, Period], SeriesVolume, name="sma")

pset.addEphemeralConstant("StdDev", lambda: StdDev(random.uniform(1., 3.)), StdDev)
pset.addEphemeralConstant("MAMode", lambda: MAMode(random.choice(['ema', 'sma'])), MAMode) # Needs to be python executable so use "'ema'"
pset.addEphemeralConstant("BBOut", lambda: BBOut(random.choice(['bbl', 'bbm', 'bbu', 'bbb', 'bbp'])), BBOut)
for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(bbands, [combo, Period, StdDev, MAMode, BBOut], SeriesIndicator, name="bbands")
    pset.addPrimitive(bbands, [combo, Period, StdDev, MAMode, BBOut], SeriesFloat, name="bbands")

for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(highest, [combo, Period], SeriesIndicator, name="highest")
    pset.addPrimitive(lowest, [combo, Period], SeriesIndicator, name="lowest")
    #--
    pset.addPrimitive(highest, [combo, Period], SeriesFloat, name="highest")
    pset.addPrimitive(lowest, [combo, Period], SeriesFloat, name="lowest")
pset.addPrimitive(highest, [SeriesPrice, Period], SeriesPrice, name="highest")
pset.addPrimitive(lowest, [SeriesVolume, Period], SeriesVolume, name="lowest")


# Executing Code Block for RSI, and Others
from evoquant.indicators import COMPILE_RSI, COMPILE_ZSCORE
exec(COMPILE_RSI)
exec(COMPILE_ZSCORE)

#=======================================================================================================================
pset.addPrimitive(and_rule, [SeriesBool, SeriesBool], SeriesBool, name="and_rule")
pset.addPrimitive(and_rule3, [SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="and_rule3")
pset.addPrimitive(and_rule4, [SeriesBool, SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="and_rule4")

pset.addPrimitive(or_rule, [SeriesBool, SeriesBool], SeriesBool, name="or_rule")
pset.addPrimitive(or_rule3, [SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="or_rule3")
pset.addPrimitive(or_rule4, [SeriesBool, SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="or_rule4")

pset.addPrimitive(xor_rule, [SeriesBool, SeriesBool], SeriesBool, name="xor_rule")
pset.addPrimitive(xor_rule3, [SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="xor_rule3")
pset.addPrimitive(xor_rule4, [SeriesBool, SeriesBool, SeriesBool, SeriesBool], SeriesBool, name="xor_rule4")

pset.addPrimitive(not_rule, [SeriesBool], SeriesBool, name="not_rule")
#-----------------------------------------------------------------------------------------------------------------------
pset.addPrimitive(and2_or1, [SeriesBool]*4, SeriesBool, name="and2_or1")
pset.addPrimitive(and3_or1, [SeriesBool]*6, SeriesBool, name="and3_or1")
pset.addPrimitive(and2_or2, [SeriesBool]*6, SeriesBool, name="and2_or2")
pset.addPrimitive(and3_or2, [SeriesBool]*9, SeriesBool, name="and3_or2")
pset.addPrimitive(and2_or3, [SeriesBool]*8, SeriesBool, name="and2_or3")
pset.addPrimitive(and3_or3, [SeriesBool]*12, SeriesBool, name="and3_or3")
#-----------------------------------------------------------------------------------------------------------------------
pset.addPrimitive(or2_and1, [SeriesBool]*4, SeriesBool, name="or2_and1")
pset.addPrimitive(or3_and1, [SeriesBool]*6, SeriesBool, name="or3_and1")
pset.addPrimitive(or4_and1, [SeriesBool]*8, SeriesBool, name="or4_and1")
pset.addPrimitive(or2_and2, [SeriesBool]*6, SeriesBool, name="or2_and2")
pset.addPrimitive(or3_and2, [SeriesBool]*9, SeriesBool, name="or3_and2")
pset.addPrimitive(or4_and2, [SeriesBool]*12, SeriesBool, name="or4_and2")
pset.addPrimitive(or2_and3, [SeriesBool]*8, SeriesBool, name="or2_and3")
pset.addPrimitive(or3_and3, [SeriesBool]*12, SeriesBool, name="or3_and3")
pset.addPrimitive(or4_and3, [SeriesBool]*16, SeriesBool, name="or4_and3")
#-----------------------------------------------------------------------------------------------------------------------
for combo in generate_2permutations([SeriesFloat, SeriesIndicator, SeriesPrice]):
    pset.addPrimitive(cross_above_rule, [combo[0], combo[1]], SeriesBool, name="cross_above_rule")
    pset.addPrimitive(cross_below_rule, [combo[0], combo[1]], SeriesBool, name="cross_below_rule")
    pset.addPrimitive(is_above_rule, [combo[0], combo[1]], SeriesBool, name="is_above_rule")
    pset.addPrimitive(is_below_rule, [combo[0], combo[1]], SeriesBool, name="is_below_rule")

pset.addEphemeralConstant("IncrDecrNBars", lambda: IncrDecrNBars(random.randint(2, 7)), IncrDecrNBars)
for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(is_incr_n_bars_rule, [combo, IncrDecrNBars], SeriesBool, name="is_incr_n_bars_rule")
    pset.addPrimitive(is_decr_n_bars_rule, [combo, IncrDecrNBars], SeriesBool, name="is_decr_n_bars_rule")

pset.addEphemeralConstant("HighLowNBars", lambda: HighLowNBars(random.randint(2, 30)), HighLowNBars)
for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]:
    pset.addPrimitive(is_highest_n_bars_rule, [combo, HighLowNBars], SeriesBool, name="is_highest_n_bars_rule")
    pset.addPrimitive(is_lowest_n_bars_rule, [combo, HighLowNBars], SeriesBool, name="is_lowest_n_bars_rule")

pset.addEphemeralConstant("DayOfWeek", lambda: DayOfWeek(random.randint(1, 5), min_val=1, max_val=7), DayOfWeek)
# pset.addEphemeralConstant("MonthInYear", lambda: MonthInYear(random.randint(1, 12), min_val=1, max_val=12), MonthInYear)
# pset.addEphemeralConstant("HourInDay", lambda : HourInDay(random.randint(0, 23)), HourInDay)
pset.addPrimitive(day_of_week_rule, [SeriesDate, DayOfWeek], SeriesBool, name="day_of_week_rule")
# pset.addPrimitive(month_in_year_rule, [SeriesDate, MonthInYear], SeriesBool, name="month_in_year_rule")
# pset.addPrimitive(hour_in_day_rule, [SeriesDate, HourInDay], SeriesBool, name="hour_in_day_rule")
# pset.addPrimitive(hour_in_day_ge_rule, [SeriesDate, HourInDay], SeriesBool, name="hour_in_day_ge_rule")
# pset.addPrimitive(hour_in_day_le_rule, [SeriesDate, HourInDay], SeriesBool, name="hour_in_day_le_rule")

pset.addEphemeralConstant("Quantile", lambda: Quantile(round(random.uniform(0.01, 0.99), 4)), Quantile)
for combo in [SeriesFloat, SeriesIndicator, SeriesPrice]: # Stationarity-agnostic
    pset.addPrimitive(series_above_quantile_rule, [combo, Quantile, Period], SeriesBool, name="series_above_quantile_rule")
    pset.addPrimitive(series_below_quantile_rule, [combo, Quantile, Period], SeriesBool, name="series_below_quantile_rule")
    pset.addPrimitive(series_cross_above_quantile_rule, [combo, Quantile, Period], SeriesBool, name="series_cross_above_quantile_rule")
    pset.addPrimitive(series_cross_below_quantile_rule, [combo, Quantile, Period], SeriesBool, name="series_cross_below_quantile_rule")

    pset.addPrimitive(series_above_ma_rule, [combo, Period], SeriesBool, name="series_above_ma_rule")
    pset.addPrimitive(series_below_ma_rule, [combo, Period], SeriesBool, name="series_below_ma_rule")
    pset.addPrimitive(series_cross_above_ma_rule, [combo, Period], SeriesBool, name="series_cross_above_ma_rule")
    pset.addPrimitive(series_cross_below_ma_rule, [combo, Period], SeriesBool, name="series_cross_below_ma_rule")

    pset.addPrimitive(series_above_shift_rule, [combo, Lag], SeriesBool, name="series_above_shift_rule")
    pset.addPrimitive(series_below_shift_rule, [combo, Lag], SeriesBool, name="series_below_shift_rule")
    pset.addPrimitive(series_cross_above_shift_rule, [combo, Lag], SeriesBool, name="series_cross_above_shift_rule")
    pset.addPrimitive(series_cross_below_shift_rule, [combo, Lag], SeriesBool, name="series_cross_below_shift_rule")
########################################################################################################################

# Run to get the classes and functions from other modules to the pset because pset does not recognize the custom classes
import importlib
import inspect
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
            pset.mapping[name] = item # Add the Classes and Functions from other modules to pset.mapping
    # print(global_symbols)
store_imported_items('..base', 'evoquant.base')
store_imported_items('..indicators', 'evoquant.indicators')
store_imported_items('..signals', 'evoquant.signals')

# pset.context.pop('__builtins__') # To remove the key-value pair '__builtins__': None, which causes the NoneType error.
# pset_context = copy.deepcopy(pset.context) # Make a Deep Copy
# print(pset_context)
# print(pset.context)
# print(pset.mapping)

# Create a Function from evo_bt backtesting engine.
test_d = copy.deepcopy(df_ohlcv)
test_d.set_index('Date', inplace=True)
test_d.index = pd.to_datetime(test_d.index)
test_d = test_d.sort_index()

# Create Toolbox Before Registering th Evaluator
from evoquant.evo_gp import evo_evaluator, PerfStats

toolbox = base.Toolbox()

perf_stats = PerfStats(("Calmar", 1.)) # Instantiate PerfStats object
perf_stats.fitness_targets = (1.,) # Set the Fitness Targets

creator.create("FitnessMulti", base.Fitness, weights=perf_stats.weights)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

from evoquant.backtest_engine.validation import linear_is_oos, multi_linear_is_oos
strat_params = dict(strategy_name="TestBot", direction="LongOnly", trade_size=50,
                    exit_encoded_entry=False, exit_after_n_bars=None, exit_after_n_days=3, exit_end_of_week=False, exit_end_of_month=False, enable_tsl=False,
                    stop_loss=tuple(), take_profit=tuple(), exit_when_pnl_lessthan=None)

evo_bt_params = dict(use_pct_rets=True, splitter_func=multi_linear_is_oos,
                     splitter_params=dict(n_splits=2, train_ratio=0.55),
                     strat_params=strat_params)

bt = Backtest(test_d, EvoStrategy, cash=10000., commission=.002, margin=0.05, trade_on_close=False, hedging=False, exclusive_orders=False)

# f2_ls = [('ISOOS', 'Calmar', 'gt', .0, {}), ('OOS', 'Sharpe', 'gt', .0, {})]
f2_ls = [('OOS', 'Sharpe', 'gt', .0, {})]
toolbox.register("evaluate", evo_evaluator, pset=pset, pset_mapping=pset.mapping, main_input=input_gp,
                 bt=bt, evo_bt_params=evo_bt_params, perf_stats=perf_stats,
                 use_filter_layer1=True, use_filter_layer2=False, filter_layer2_list=f2_ls
                 )

# Problem with 'evaluate is that it only returns the fitness value, which is a tuple of numbers we need to get a way to modify the Individual PrimitiveTree'
# We need a function to return an evaluated Individual i.e. We added/modified the .fitness.values property
from evoquant.evo_gp import eval_modifier, generate_composite_signal_root, evo_populator
toolbox.register('eval_modifier', eval_modifier, evaluator=toolbox.evaluate)

# toolbox.register("expr", generate_safe, pset=pset, min_=5, max_=50, terminal_types=terminal_types)
toolbox.register("expr", generate_composite_signal_root, pset=pset, min_=1, max_=3, terminal_types=terminal_types)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("population", evo_populator, list, toolbox.individual)
toolbox.register("compile", evo_compiler, pset=pset, p_context=pset.mapping)
# toolbox.register("mutate", evo_mutation, expr=toolbox.expr, pset=pset, mutators=[gp.mutShrink, gp.mutEphemeral, gp.mutNodeReplacement])
toolbox.register("mutate", evo_mutation, expr=toolbox.expr, pset=pset)
toolbox.register("mate", evo_cross, mutator=toolbox.mutate, termpb=0.05)
toolbox.register("select", tools.selDoubleTournament, fitness_size=4, parsimony_size=1.5, fitness_first=True)


"""Notes
In the actual main program, we structure it like

def main():
    pass

if __name__ == '__main__':
    main()
"""

if __name__ == '__main__':
    from evoquant.evo_gp import gp_main_algo_random, gp_main_algo_standard_gp, gp_main_algo_multi_islands
    import math

    POP_SIZE = 1000


    # with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = [executor.submit(toolbox.individual) for _ in range(POP_SIZE)]
    #     population = []
    #     for future in concurrent.futures.as_completed(futures):
    #         result = future.result()
    #         population.append(result)
    # executor.shutdown()
    # for i in population:
    #     print(i)

    # gp_main_res = \
    # gp_main_algo_multi_islands(toolbox, "max", perf_stats, n_demes=16, n_migrants=5, init_n_gen=7, migration_rate=5, in_population=[],
    #                            pop_size=POP_SIZE, n_children=math.floor(POP_SIZE*1.0), mut_pb=0.5, cx_pb=0.5, n_gen=25,
    #                            single_fitness_method="weighted_sum", fitness_idx=None,
    #                            gp_algo_exit_params=dict(no_evol_after_ngen=5, check_fitness_targets=True, after_n_hours=1, no_valid_pop_after_n_gen=6),
    #                            concurrent_exec_params=dict(use_concurrent=False, n_processors=16, chunksize=None)
    #                            )

    # test_pop_invalid = toolbox.population(30) # Unevaluated. But even if evaluated, some values could be np.nan
    #
    # import concurrent.futures
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)
    # toolbox.register("map", executor.map, timeout=None, chunksize=1)
    #
    # # Creating a Valid Population for Testing
    # import random
    # test_pop_valid = list(toolbox.map(toolbox.eval_modifier, toolbox.population(POP_SIZE)))
    # for ind in test_pop_valid:
    #     ind.fitness.values = (random.uniform(-0.5, 0.001),)
    #     ind.fitness_achieved = random.choice([True, False]) # Have to reset to False
    #     # print(ind.fitness, "|", type(ind), "|", ind)
    # del ind
    # executor.shutdown()  # Shutdown executor
    # toolbox.unregister("map")  # Unregister the map
    #
    gp_main_res = \
    gp_main_algo_standard_gp(toolbox, "max", perf_stats, in_population=[],
                             pop_size=POP_SIZE, n_children=math.floor(POP_SIZE*1.3), n_gen=6, mut_pb=0.4, cx_pb=0.6,
                             single_fitness_method = "weighted_sum", fitness_idx = None,
                             gp_algo_exit_params=dict(no_evol_after_ngen=4, check_fitness_targets=True, after_n_hours=0.25, no_valid_pop_after_n_gen=5),
                             concurrent_exec_params=dict(use_concurrent=True, n_processors=10, chunksize=None)
                             )

    # gp_main_res = \
    # gp_main_algo_random(toolbox, "max", perf_stats,
    #                     pop_size=POP_SIZE, n_gen=10,
    #                     single_fitness_method="weighted_sum", fitness_idx=None,
    #                     gp_algo_exit_params=dict(no_evol_after_ngen=4, check_fitness_targets=True, after_n_hours=1, no_valid_pop_after_n_gen=5),
    #                     concurrent_exec_params=dict(n_processors=16, chunksize=None, timeout=None)
    #                     )


    # temp_ls = [gp.PrimitiveTree(ind) for ind in toolbox.population(POP_SIZE)]
    temp_ls = gp_main_res
    dump_data = \
        {
            "population":temp_ls,
            "strat_config":strat_params
        }
    import pickle
    with open(r"D:\Projects\FinDashAnalytics\Data\evoquant\test_strategies.pickle", "wb") as file:
        # pickle.dump(temp_ls, file)
        pickle.dump(dump_data, file)


    # i = 1
    # temp_ls = [] # For counting how many are not the same with Individual and it's Mutation.
    #
    # import time
    # start_time = time.time()
    #
    # pop_size = 1000
    # import math
    #
    # n_processes = 16
    # chunksize = math.floor(pop_size/n_processes)
    #
    # # pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=pop_size)
    # # toolbox.register("map", pool.map, chunksize=chunksize) # Have to create the map inside if __name__ == '__main__'
    # # apply_bt_pop = toolbox.map(toolbox.evaluate, toolbox.population(pop_size))
    #
    # import concurrent.futures # This is better package than multiprocessing and scoop as it can handle NoneTypes.
    # with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor: # This will be used in the main GP Algorithm for Processing multiple Islands.
    #     apply_bt_pop = [i for i in executor.map(toolbox.evaluate, toolbox.population(pop_size), chunksize=chunksize, timeout=None)]
    #     executor.shutdown()
    # assert len(apply_bt_pop) == pop_size
    # print("Length of Evaluated Backtest Population:", len(apply_bt_pop))
    #
    # # sub_apply_bt_pop = [tup for tup in apply_bt_pop if not (np.isnan(tup[0]) and (np.isnan(tup[1])))]
    # sub_apply_bt_pop = [tup for tup in apply_bt_pop if is_fitness_tuple_valid(tup)]
    # assert len(apply_bt_pop) >= len(sub_apply_bt_pop)
    # print("Length of Valid Sub-population:", len(sub_apply_bt_pop))
    #
    # for i in sub_apply_bt_pop: print(i)
    #
    # end_time = time.time()
    # print(f"Time to Evaluate the Fitness of Whole Population of {pop_size}:", end_time - start_time, "seconds")
    # del start_time, end_time


    # while True:
    #     print("Iteration: ", i)
    #     try:
    #         expr = generate_safe(pset=pset, min_=50, max_=100, terminal_types=terminal_types)
    #         tree = gp.PrimitiveTree(expr)
    #
    #         # print("Expr: ", expr)
    #         print("Expr Len: ", len(expr))
    #         print("Tree: ", tree)
    #
    #         # func_compiler = gp.compile(tree, pset)
    #         func_compiler = evo_compiler(tree, pset, pset.mapping)
    #         res = func_compiler(*input_gp)
    #
    #         print("Fitness: ", toolbox.evaluate(tree)[0])
    #         print(res.series)
    #
    #
    #
    #         # # Testing Crossover
    #         # while True:
    #         #     parent1 = copy.deepcopy(toolbox.individual())
    #         #     parent2 = copy.deepcopy(toolbox.individual())
    #         #     if str(parent1) != str(parent2): # Exit this nested loop if the parents are
    #         #         break
    #         # print(f"Parent1: {parent1}")
    #         # print(f"Parent2: {parent2}")
    #         # child1, child2 = toolbox.mate(parent1, parent2)
    #         # print(f"Child1: {child1}")
    #         # print(f"Child2: {child2}")
    #         # print(f"Child1 Fitness: {toolbox.evaluate(child1)[0]}")
    #         # print(f"Child2 Fitness: {toolbox.evaluate(child2)[0]}")
    #
    #         # # Testing Mutations
    #         # mut_ind = copy.deepcopy(toolbox.individual())
    #         # print("Individual to be mutated: ", mut_ind)
    #         # print("Type: ", type(mut_ind))
    #         # print("Fitness: ", toolbox.evaluate(mut_ind)[0])
    #         # print("Length: ", len(mut_ind))
    #         # print("\n")
    #         # try:
    #         #     mutated_ind = toolbox.mutate(copy.deepcopy(mut_ind))[0] #Tuple[gp.PrimitveTree,None]
    #         #     print("Mutation: ", mutated_ind)
    #         #     print("Fitness: ", toolbox.evaluate(mutated_ind)[0])
    #         #     print("Length: ", len(mutated_ind))
    #         # except Exception as error:
    #         #     print(type(error))
    #         #     print(error)
    #         #     continue
    #         # are_same = str(mut_ind) == str(mutated_ind)
    #         # if are_same:
    #         #     temp_ls.append(0)
    #         # else:
    #         #     temp_ls.append(1)
    #         # print("Are they the same?  Ans:", are_same)
    #
    #         if i>=500:
    #             break
    #     except IndexError as error:
    #         # e.g. An error occurred: The gp.generate function tried to add a terminal of type '<class 'evoquant.base.SeriesBool'>', but there is none available.
    #         if 'SeriesBool' in str(error):
    #             print("We found error but skipped")
    #             print(error)
    #             break
    #         else:
    #             print(error)
    #             break
    #     except TypeError as error:
    #         print(error)
    #         break
    #     except Exception as error:
    #         print(type(error))
    #         print(error)
    #         break
    #     finally:
    #         i += 1
    #         print("=" * 200)
    # del i
    # # print("Ratio of Instances that are not the same: ", (sum(temp_ls)/len(temp_ls))*100, "%")