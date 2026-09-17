import inspect, random, sys, os, pickle, copy, pickle, datetime, uuid
from itertools import combinations_with_replacement, permutations
from functools import partial

import deap.tools
from deap import gp

import quantstats
import empyrical
import pandas as pd
import numpy as np

from backtesting import Backtest

from typing import List, Union, Tuple

from evoquant.base import SeriesBool


def _measure_execution_time(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper

def _save_result_to_pickle(directory, file_extension='pickle'):
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            result = func(*args, **kwargs)

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Generate a unique filename based on timestamp and UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = os.path.join(directory, fr"""strategies_{timestamp}_{unique_id}.{file_extension}""")

            # Save the result as a pickle file
            with open(filename, "wb") as file:
                pickle.dump(result, file)

            return result
        return wrapper_func
    return decorator_func


def evo_compiler(expr, pset, p_context):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :param p_context: Dictionary containing the mapping from string to Primitives, Terminals, and Custom Types.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    # This section is a stripped version of the lambdify
    # function of SymPy 0.6.6.
    args = ",".join(arg for arg in pset.arguments)
    code = "lambda {args}: {code}".format(args=args, code=code)
    return eval(code, p_context, {})

# TODO: Find a solution where it can generate a safe and valid PrimitiveTree based on the given height constraints.
def generate_safe(pset, min_, max_, terminal_types, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if type_ in terminal_types:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if inspect.isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                # Might not be respected if there is a type without terminal args
                if height <= depth or (depth >= min_ and random.random() < pset.terminalRatio):
                    primitives_with_only_terminal_args = [p for p in pset.primitives[type_] if
                                                          all([arg in terminal_types for arg in p.args])]
                    if len(primitives_with_only_terminal_args) == 0:
                        prim = random.choice(pset.primitives[type_])
                    else:
                        prim = random.choice(primitives_with_only_terminal_args)
                else:
                    prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

def generate_composite_signal_root(pset, min_, max_, terminal_types, type_=None):
    """This function ensures that the root is a "Composite Signal" Primitive"""
    if type_ is None:
        type_ = pset.ret # Return type of the 'main' pset
    expr = [] # List[Terminal|Primitive]
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    i = 0
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if type_ in terminal_types:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if inspect.isclass(term):
                term = term()
            expr.append(term)
        else:
            try:

                if i == 0: # At the Root
                    primitives_with_seriesbool_args = [p for p in pset.primitives[type_] if all([arg == SeriesBool for arg in p.args])]
                    prim = random.choice(primitives_with_seriesbool_args)
                # Might not be respected if there is a type without terminal args
                elif height <= depth or (depth >= min_ and random.random() < pset.terminalRatio):
                    primitives_with_only_terminal_args = [p for p in pset.primitives[type_] if all([arg in terminal_types for arg in p.args])]
                    if len(primitives_with_only_terminal_args) == 0: # i.e. Its Terminal
                        prim = random.choice(pset.primitives[type_])
                    else:
                        prim = random.choice(primitives_with_only_terminal_args)
                else:
                    prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
        i += 1
    return expr


def generate_3permutations(my_list):
    # Generate 3-tuple combinations with replacement, including element with itself
    combinations_list = combinations_with_replacement(my_list, 3) #list(combinations_with_replacement(my_list, 3))

    # Generate permutations of 3-tuple combinations
    permutations_list = [(x, y, z) for x, y, z in combinations_list] + \
                        [(x, z, y) for x, y, z in combinations_list if x != y] + \
                        [(y, x, z) for x, y, z in combinations_list if x != z and y != z] + \
                        [(y, z, x) for x, y, z in combinations_list if x != y] + \
                        [(z, x, y) for x, y, z in combinations_list if x != z and y != z] + \
                        [(z, y, x) for x, y, z in combinations_list if x != y]

    # Print the permutations
    return (permutation for permutation in permutations_list)

def generate_2permutations(my_list):
    # Generate 2-tuple combinations with replacement, including element with itself
    combinations_list = combinations_with_replacement(my_list, 2) #list(combinations_with_replacement(my_list, 2))

    # Generate permutations of 2-tuple combinations
    permutations_list = [(x, y) for x, y in combinations_list] + [(y, x) for x, y in combinations_list if x != y]

    return (permutation for permutation in permutations_list)

def is_fitness_tuple_valid(in_tup:Union[Tuple,List]) -> bool:
    """Return True if all the elements in a tuple|list of fitnesses are valid.
    """
    import numpy as np
    #
    out_ls = []
    for i in in_tup:
        if not np.isnan(i) and not np.isinf(i) and i is not None and i != "" and type(i) != type(None) and isinstance(i, (float, int)):
            out_ls.append(True)
        else:
            out_ls.append(False)
    return all(out_ls)

def evo_populator(container, ind_generator:callable, n:int) -> list:
    import concurrent.futures
    import multiprocessing

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(ind_generator) for _ in range(n)]
    executor.shutdown()
    return container(future.result() for future in futures)

def evo_mutation(individual:gp.PrimitiveTree, expr:callable, pset:gp.PrimitiveSetTyped,
                 mutators=[gp.mutShrink, gp.mutUniform, gp.mutNodeReplacement, gp.mutEphemeral, gp.mutInsert]
                 ):

    ind = copy.deepcopy(individual) # Deepcopy the individual to be mutated. We don't want to modify the original input.

    while True:
        # Randomly select mutation function from deap gp
        _mutator = random.choice(mutators)
        if _mutator is gp.mutShrink:
            mutator = _mutator
        if _mutator is gp.mutUniform:
            mutator = partial(_mutator, expr=expr, pset=pset)
        if _mutator is gp.mutNodeReplacement:
            mutator = partial(_mutator, pset=pset)
        if _mutator is gp.mutEphemeral:
            mutator = partial(_mutator, mode=random.choice(['one', 'all']))
        if _mutator is gp.mutInsert:
            mutator = partial(_mutator, pset=pset)

        # We only exit the loop when we generate a mutation of the individual that is different.
        try: # Some mutator could result in an error e.g. IndexError
            mutated = mutator(ind)[0]
            if str(individual) != str(mutated):
                break
        except Exception as err:
            # print(f"We tried to use the function {_mutator.__name__}, but it produces an error {type(err)}.")
            continue

    # Mutated Tuple[gp.PrimitiveTree,None]
    return mutated,

def evo_cross(parent1:gp.PrimitiveTree, parent2:gp.PrimitiveTree, mutator:callable, termpb:float=0.5, *args, **kwargs):
    # Remind that mutator:(gp.PrimitiveTree,...) -> Tuple[gp.PrimitiveTree,None]
    if str(parent1) == str(parent2): # Mutate the parents if they are equal.
        are_parents_mutated = True
        while True:
            p1 = mutator(copy.deepcopy(parent1), *args, **kwargs)[0]
            p2 = mutator(copy.deepcopy(parent2), *args, **kwargs)[0]
            if str(p1) != str(p2):
                break
    else:
        are_parents_mutated = False
        p1 = copy.deepcopy(parent1)
        p2 = copy.deepcopy(parent2)

    while True:
        _crossover = random.choice([gp.cxOnePoint, gp.cxOnePointLeafBiased]) # Randomly select crossover method
        if _crossover is gp.cxOnePoint:
            crossover = gp.cxOnePoint
        if _crossover is gp.cxOnePointLeafBiased:
            crossover = partial(gp.cxOnePointLeafBiased, termpb=termpb)

        child1, child2 = crossover(copy.deepcopy(p1), copy.deepcopy(p2)) # Try to apply the crossover method

        if (str(child1) in [str(p1), str(p2)]) and (str(child2) in [str(p1), str(p2)]):
            # If one of the children is equal to one of the parents, then return the mutations of the parents
            if are_parents_mutated:
                return p1, p2
            else:
                return mutator(copy.deepcopy(p1), *args, **kwargs)[0], mutator(copy.deepcopy(p2), *args, **kwargs)[0]
        else:
            return child1, child2

def evo_varor(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in toolbox.select(population, 2)] # We use our select method rather than pure random
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    return offspring


class PerfStats:
    """This will be the generalized solution for handling multiple fitness functions for GP.
    Parameters:
    in_fits_weights : List[Tuple[Str, Float|Int]]
        List of 2-tuples containing the Valid Fitness function name (e.g. "Sharpe") and appropriate weights (e.g. -1. tos 1.).

    Properties:
    n_fitness : Int
        Number of Fitness Functions.
        This is always fixed
    fitness_names : Tuple[Str]
        List of Fitness Functions / Performance Statistics.
        This is always fixed.
    weights : Tuple[Float|Int]
        Weights given to the fitness/performance statistics, with signs indicating the optimization direction for gp.
        This will be an input for deap creator e.g. creator.create("FitnessMulti", base.Fitness, weights = PerfStats.weights).
        This is always fixed
    invalid_values : Tuple[np.nan]
        This will be used as default output of evo_evaluator when one of thr fitness values is invalid. By invalid, meaning
        np.nan, np.inf, NonType, "", etc. This is always fixed.
    fitness_values : Tuple[Float|Int] | Tuple[np.nan] | Tuple[Float|Int|np.nan]
        Fitness Values when rets and df_trades is set from set_fitness_required_args. The method will invoke _calc_fitness_values and automatically sets fitness values.
        This is dynamically changed everytime the evaluator is used.
    _PS : Dict[Str->Callable]
        This dictionary contains all available performance statistics and their corresponding functions from empyrical, quantstats, or custom.
    fitness_targets:Tuple[float]
        Tuple of Float|Int containing the fitness targets to be used as exit criterion in a gp main algorithm.

    Methods:
    from_lists(fit_names:List[Str], weights:List[Float|Int]) -> void
        Class method for alternative way to instantiate PerfStats class.
    (Deprecated) set_return_series(rets:pd.Series) -> void
            This method can be used to set the return series because there are cases when the fitness functions only require the return series.
    (Deprecated) set_trades_df(df_trades:pd.DataFrame) -> void
            This method can be used to set the trades dataframe because there are cases when the fitness functions only require the trades dataframe.
    set_fitness_required_args(rets:pd.Series, df_trades:pd.DataFrame)
        This method can be used to set both the return series and trades dataframe because there are cases when some fitness functions requires one of them.
        For fitness calculation, we need In-Sample return series and trades dataframe!!!
    get_fitness_func(fitness_name:Str) -> Callable
    (Deprecated) calc_fitness(fitness_name:Str, pd.Series|pd.DataFrame, *args, **kwargs) -> Float
    _calc_fitness_values() -> Tuple[Float|Int] | Tuple[np.nan]
        Used as the output for evo_evaluator. There's possibility that this method returns a Tuple[np.nan] either because
        one of the fitness calculation throws a ZeroDivisionError or OverflowError or other runtime errors from other
        packages (empyrical or quantstats or backtesting). Special Case when there's only 1 fitness given. Output should still be
        a tuple, so check with n_fitness parameter.


    Constructions:
    PERF_STATS = PerfStats(("Sharpe", 1), ("Sortino", 1), ("MaxDD", -1), ...) | PerfStats(*List[Tuple[Str, Float|Int]])
    PERF_STATS = PerfStats.from_lists(["Sharpe", "Sortino", "MaxDD"], [1, 1, -1])
    """
    import operator

    # TODO: Make a private module for global variable performance statistics function dictionary and required input dictionary. Same with 2nd Filter Layer.
    _OP = {"gt": operator.gt,
           "ge": operator.ge,
           "lt": operator.lt,
           "le": operator.le,
           "eq": operator.eq,
           "ne": operator.ne
           }

    _PS = {"Sharpe":quantstats.stats.sharpe,
           "Calmar":quantstats.stats.calmar,
           "Sortino":quantstats.stats.sortino,
           "CAGR/AvgDD": lambda rets: quantstats.stats.cagr(rets) / quantstats.stats.to_drawdown_series(rets).abs().mean(),
           "Stability":empyrical.stability_of_timeseries,
           # "AvgMonthlyReturns": lambda rets: quantstats.stats.monthly_returns(rets, eoy=False)['Month'].mean(),
           "Volatility":quantstats.stats.volatility,
           "VaR":quantstats.stats.value_at_risk,
           "CVaR":quantstats.stats.conditional_value_at_risk,
           "MaxDD":empyrical.max_drawdown,
           "AvgDD":lambda rets: quantstats.stats.to_drawdown_series(rets).mean(),
           "MaxDD_Duration":lambda rets: quantstats.stats.drawdown_details(quantstats.stats.to_drawdown_series(rets))['days'].max(),
           "Avg$PnL":lambda df: df['PnL'].mean(),
           "Avg$Loss": lambda df: df['PnL'].where(df['PnL'] < 0.).mean(),
           "Avg$Profit": lambda df: df['PnL'].where(df['PnL'] > 0.).mean(),
           "NumberOfTrades":lambda df: df.shape[0],
           "Total$PnL": lambda df: df['PnL'].sum(),
           "Max$Loss":lambda df: df['PnL'].min()
           }

    _PS_IN = \
        {"Sharpe": 0,
         "Calmar": 0,
         "Sortino": 0,
         "CAGR/AvgDD": 0,
         "Stability": 0,
         # "AvgMonthlyReturns": 0,
         "Volatility": 0,
         "VaR": 0,
         "CVaR": 0,
         "MaxDD": 0,
         "AvgDD": 0,
         "MaxDD_Duration": 0,
         "Avg$PnL": 1,
         "Avg$Loss": 1,
         "Avg$Profit": 1,
         "NumberOfTrades": 1,
         "Total$PnL": 1,
         "Max$Loss": 1
         }

    # These parameters are dynamically changed.
    rets:pd.Series = None
    df_trades:pd.DataFrame = None
    fitness_values:tuple = None
    _fitness_targets:Union[List,Tuple] = None

    def __init__(self, *in_fits_weights):
        assert len(in_fits_weights) >= 1
        self.n_fitness = len(in_fits_weights)
        self.fitness_names, self.weights = zip(*in_fits_weights)
        assert isinstance(self.fitness_names, tuple) and isinstance(self.weights, tuple)

        for name in self.fitness_names:
            if (name not in self._PS.keys()) or (not isinstance(name, str)):
                print(f"Valid Fitness Names: \n{self._PS}")
                raise ValueError("The Fitness Names has to be valid.")
        for w in self.weights:
            if (not isinstance(w, (float, int))):
                raise ValueError("Weight has to be Float or Integer.")
            if (w > 1.) or (w < -1.) or (w == 0):
                raise ValueError("Weight Values has to be in closed interval [-1, 1] and cannot be equal to zero.")

        self.invalid_values = (np.nan,)*self.n_fitness # Tuple([np.nan for _ in range(self.n_fitness)])

    @classmethod
    def from_lists(cls, fitness_names:List[str], weights:List[Union[float,int]]):
        return cls(*list(zip(fitness_names, weights)))

    @classmethod
    def get_fitness_func(cls, fitness_name:str) -> callable:
        return cls._PS[fitness_name]

    @classmethod
    def get_valid_fitness_names(cls):
        return list(cls._PS.keys())

    @property
    def fitness_targets(self):
        return self._fitness_targets

    @fitness_targets.setter
    def fitness_targets(self, in_fitness_targets:Union[List, Tuple]):
        """This method need to be set outside of gp main algorithms. This should be set when it's instantiated. Be default it's equal to None.

        Example:
        perf_stats = PerfStats(('Sharpe', 1.),('MaxDD',-1.))
        perf_stats.fitness_targets = (1.2, -.05)
        """
        if not isinstance(in_fitness_targets, (list,tuple)) or len(in_fitness_targets) != self.n_fitness or any([not isinstance(i, (float,int)) for i in in_fitness_targets]):
            raise ValueError("The Fitness Target(s) is Invalid.")
        self._fitness_targets = in_fitness_targets

    def set_fitness_required_args(self, in_rets:pd.Series, in_df_trades:pd.DataFrame):
        self.rets = copy.deepcopy(in_rets)
        self.df_trades = copy.deepcopy(in_df_trades)
        # Invoke _calc_fitness_values
        # Check if all fitness values are valid, otherwise set the invalid tuple self.invalid_values
        out_vals = self._calc_fitness_values() #List[Float|Int|np.nan]
        if is_fitness_tuple_valid(out_vals):
            self.fitness_values = tuple(out_vals)
        else:
            self.fitness_values = copy.deepcopy(self.invalid_values)

    def _calc_fitness_values(self) -> List[float]: # In Reality, List[Float|Int|np.nan]
        """
        Use n_fitness and _PS["<Fitness Name>"]->Callable
        """
        temp_fitness_values = [] # To be converted into tuple
        for f_name in self.fitness_names:
            try:
                if self._PS_IN[f_name] == 0: # Fitness Function Requires Return Series
                    f = self._PS[f_name](self.rets)
                if self._PS_IN[f_name] == 1:  # Fitness Function Requires Trades DataFrame
                    f = self._PS[f_name](self.df_trades)
            except: # If we catch any error from calculating a Fitness, append np.nan
                temp_fitness_values.append(np.nan)
            else: # If it didn't throw any error, we still need to check if the value is valid, otherwise append np.nan
                if not np.isnan(f) and not np.isinf(f) and f is not None and f != "" and type(f) != type(None) and isinstance(f, (float, int)):
                    temp_fitness_values.append(float(f)) # Need to convert to float, otherwise it will include float and int in tuple which causes TypeError.
                else:
                    temp_fitness_values.append(np.nan)
        return temp_fitness_values

from evoquant.backtest_engine.evo_bt import evo_backtester, evo_filter_layer1, evo_filter_layer2

def evo_evaluator(individual:gp.PrimitiveTree, pset:gp.PrimitiveSetTyped, pset_mapping:dict, main_input:tuple,
                  bt:Backtest, evo_bt_params:dict, perf_stats:PerfStats,
                  use_filter_layer1:bool=True, use_filter_layer2:bool=False, filter_layer2_list:List[Tuple[str,str,str,Union[float,int],dict]]=[]
                  ) -> tuple:
    """This function will be the bridge between the backtesting engine and GP. This will be used DEAP toolbox
    toolbox.register(evo_evaluator, *args, **kwargs)

    Return : Tuple[float, float, ...]
        These represent the tuple of different fitness performance stats. Some could be Maximization or Minimization Problems.

    Returning the Fitnesses Tuples. Note that we could be Calculating multiple Fitness Values with different weights from Float[-1., 1.]
    (+Sharpe, +Calmar, +Sortino, -MaxDD, -AvgLoss) : (1, 1, 1, -1, -1) = Weights

    The Direction of Optimization (Max/Min) must be specified by the user.

    In DEAP, we can have FitnessMax, FitnessMin, and FitnessMulti.

    Warnings:
    - Some fitness value can be None, np.nan, or np.inf.
    - Some Performance Stats can throw errors such as ZerDivisionError.
    - For FitnessMulti e.g. w=(1, -1), there will be cases where the return is (0.0,) and (np.nan, np.nan) and (np.inf, <ValidValue>)
        But the output will depend on the Fitness Performance Stats used.

    Note: Fitness Value has to be calculated on In-Sample Only. Thus, IS-OOS parameters has to be specified and given.
    Note: We will only apply the 1st & 2nd Filter Layers.
        1st Layer - Filtering out the anomalies (e.g. 1 Trade, No Trade, All True/False in SeriesBool, etc.)
        2nd Layer - Filtering in Individuals who pass the conditions for IS, OOS, and ISOOS.
        3rd Layer - Not included as this will be computationally expensive. This is done when the main GP loop is finished.
    Note: Depending on the performance stat calculator (QuantStats or Empyrical), MaxDD values may be negative.
    Note: For Individual that are anomalies, it may return a null or np.nan values in fitness(es). Which has to be part of rejection.
    """
    func_compiler = evo_compiler(individual, pset, pset_mapping) # Compile the expression and get the unevaluated tree
    ser_bool = func_compiler(*main_input)  # -> SeriesBool, Compiler

    # Step: Apply 1st Filter Layer.
    # Step: Apply 2nd Filter Layer. (Warning: The GP Algorithm may have trouble finding profitable strategies)
    # Step: If Filters passed, return In-sample Fitness(es).

    # When we run this, we are basically running backtesting.Backtest.run, which will perform some stats, some may cause errors similar to
    # to when calculating performance stats, ZeroDivisionError or OverflowError.

    # Run evo_backtester function and try to catch all possible errors.

    # direction = evo_bt_params["direction"]

    try:
        evo_bt_res = evo_backtester(ser_bool, bt, **evo_bt_params)
    except ZeroDivisionError as err:
        print(str(err))
        return perf_stats.invalid_values
    except OverflowError as err:
        print(str(err))
        return perf_stats.invalid_values
    except Exception as err:
        raise Exception(str(err))

    # Calculate the In-Sample Fitness Values in PerfStats Object
    # Remember that PerfStats needs to be instantiated outside of this function.
    perf_stats.set_fitness_required_args(evo_bt_res["IS"][0], evo_bt_res["IS"][1]) # Modify Return Series, Trades DataFrame, and Fitness Values in perf_stats for the individual

    if use_filter_layer1:
        # Apply 1st Filter Layer for IS
        if not evo_filter_layer1(evo_bt_res['IS'][0], evo_bt_res['IS'][1]):
            return perf_stats.invalid_values
        # Apply 1st Filter Layer for OOS
        if not evo_filter_layer1(evo_bt_res['OOS'][0], evo_bt_res['OOS'][1]):
            return perf_stats.invalid_values

    if use_filter_layer2:
        if len(filter_layer2_list) == 0: raise ValueError("List of Layer 2 filters must be given.")
        filter_layer2 = evo_filter_layer2(evo_bt_res, filter_layer2_list)
        if not filter_layer2:
            return perf_stats.invalid_values

    # Fitness Evaluation
    out_fitness_values = copy.deepcopy(perf_stats.fitness_values)

    # # Modify the given Individual gp.PrimitiveTree. In GP Main Algo (Random Method), we have created a separate function for modifying this.
    # individual.fitness.values = copy.deepcopy(perf_stats.fitness_values)

    # Check if the Individual's Fitness Value(s) has satisfied the Fitness Target(s), if given.
    if perf_stats.fitness_targets != None:
        # The perf_stats.fitness_targets has been given and therefore can be compared.
        # Fitness Values has been processed by perf_stats, its either valid or invalid.
        # all([tup for tup in zip(perf_stats.weights, out_fitness_values)])
        if all([np.isnan(f) for f in out_fitness_values]):
            individual.fitness_achieved = False
            return out_fitness_values
        else: # If the fitness values are valid
            check_ls=[]
            for tup in zip(perf_stats.fitness_targets, perf_stats.weights, out_fitness_values):
                if tup[1]>0: # "max"
                    check_ls.append(tup[2] > tup[0])
                else: # "min"
                    check_ls.append(abs(tup[2]) < abs(tup[0]))
            if all(check_ls):
                individual.fitness_achieved = True
            else:
                individual.fitness_achieved = False
            return out_fitness_values
    else:
        return out_fitness_values

def transform_to_single_objective(fitness_vals:Tuple[float], weights:Tuple[float], method="weighted_sum", fitness_idx=None) -> float:
    if len(fitness_vals) != len(weights):
        raise ValueError("Length of fitness values and weights must be the same.")

    if not isinstance(method, str) or method not in ["weighted_sum", "tchebycheff", "select_fitness"]:
        raise ValueError("method must be either 'weighted_sum' or 'tchebycheff'")

    # Normalize the objective values to the range [0, 1]
    normalized_values = [(val - min(fitness_vals)) / (max(fitness_vals) - min(fitness_vals)) for val in fitness_vals]

    if method == "weighted_sum":
        # Calculate the weighted sum
        single_obj_value = sum(weight * normalized_value for weight, normalized_value in zip(weights, normalized_values))
        return single_obj_value
    if method == "tchebycheff":
        # Calculate the maximum weighted value
        max_weighted_value = max(weight * normalized_value for weight, normalized_value in zip(weights, normalized_values))
        return max_weighted_value

    if method == "select_fitness":
        if fitness_idx is None or not isinstance(fitness_idx,int):
            raise ValueError("select_fitness method requires a valid index. Take a look at the how many fitness functions to optimize.")
        if fitness_idx < 0 or fitness_idx >= len(weights):
            raise ValueError("fitness_idx must be within the index range of the length of the given fitness values or weights.")
        return fitness_vals[fitness_idx]

def eval_modifier(individual:gp.PrimitiveTree, evaluator:callable):
    """Returns the Individual with modified parameter .fitness.values"""
    # We are not going to use copy.deepcopy()
    individual.fitness.values = evaluator(individual) # Fitness Values may or may not be valid.
    return individual # Return the 'evaluated'/modified Individual

from deap import base
# @_measure_execution_time
def gp_main_algo_random(toolbox:base.Toolbox, optim_direction:str, perf_stats:PerfStats,
                        pop_size:int=100, n_gen=25,
                        single_fitness_method:str="weighted_sum", fitness_idx=None,
                        gp_algo_exit_params=dict(no_evol_after_ngen=10, check_fitness_targets=True, after_n_hours=24, no_valid_pop_after_n_gen=5),
                        concurrent_exec_params=dict(n_processors=None, chunksize=None, timeout=None)
                        ) -> Union[List[gp.PrimitiveTree], List[None]]:
    """This function is the implementation for the gp main algorithm. This uses random generation method without any evolution.

    Parameters:
    toolbox:base.Toolbox, (Required)
        base.Toolbox Instance from DEAP package. This must have all registered Objects, pset, everything.
    optim_direction:Str, (Required)
        Direction of the Optimization. But this will also depend on the supplied PerfStats object weights.

    Return:
    optimal_strategies : List[gp.PrimitiveTree]
        This list contains optimal evaluated gp.PrimitiveTree with valid fitness values.

    Notes:
    - This method must be run within the if __name__ == '__main__'.
    - PerfStats instance will be useful for algo exit criterion when using a specific list of fitness functions.
    """
    import math
    import concurrent.futures
    import multiprocessing

    if concurrent_exec_params["n_processors"] is None: # Check Number of Logical Processors
        concurrent_exec_params["n_processors"] = multiprocessing.cpu_count()
    if isinstance(concurrent_exec_params["n_processors"], int) and concurrent_exec_params["n_processors"] > multiprocessing.cpu_count():
        raise ValueError("n_processors cannot be more than the cpu's logical processors.")

    if concurrent_exec_params["chunksize"] is None: # Check Chunk Size
        concurrent_exec_params["chunksize"] = math.floor(pop_size/concurrent_exec_params["n_processors"])
    assert isinstance(concurrent_exec_params["chunksize"], int) and concurrent_exec_params["chunksize"] < pop_size

    # # Update n_generations from given n_gen. Redundant but for consistency of exit checks.
    # gp_algo_exit_params["n_generations"] = n_gen

    if len(perf_stats.weights) == 1:
        is_single_fitness = True # Set is_single_fitness. To be used later.
        if perf_stats.weights[0] > 0:
            optim_direction = "max"
        if perf_stats.weights[0] < 0:
            optim_direction = "min"
    else: # Using Multi-Objective/Fitness Functions
        is_single_fitness = False # Set is_single_fitness. To be used later.
        # Check if objective are maximization or minimization based on weights.
        if all([(w>0) for w in perf_stats.weights]):
            optim_direction = "max"
        elif all([(w<0) for w in perf_stats.weights]):
            optim_direction = "min"
        else: # Mix Direction. Which would also depend on the input of the user.
            if optim_direction is None or optim_direction not in ['max', 'min']: # Check if optim_direction is given.
                raise ValueError("Optimization Direction must be given. Either 'max' or 'min'.")

    chunk_size = concurrent_exec_params["chunksize"]

    # Start timing for after_n_hours exit criterion
    import time
    start_time = time.time()
    maximum_duration = gp_algo_exit_params["after_n_hours"] * 60 * 60  # after_n_hours hours in seconds

    print("Generating and Evaluating Generation Zero")
    print("\n")
    # Evaluate the initial pop and store the valid individuals in a list called subpop:List[gp.PrimitiveTree]
    pop = toolbox.population(pop_size)  # Create a list of unevaluated gp.PrimitiveTree. This list will be modified on runtime.
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrent_exec_params["n_processors"]) as executor:
        subpop = [ind for ind in executor.map(toolbox.eval_modifier, pop, chunksize=chunk_size, timeout=concurrent_exec_params["timeout"])]
        executor.shutdown()
    subpop = [ind for ind in subpop if is_fitness_tuple_valid(ind.fitness.values)]
    subpop_size = len(subpop)
    print("Valid Size for Generation 0 is", subpop_size)
    print("\n")
    for _ in subpop:
        print(_.fitness.values,"|", _)

    # For the exit when there is no valid population after n generations.
    if subpop_size == 0:
        count_no_valid_pop = 1
    else:
        count_no_valid_pop = 0

    # For exit when there is no improvement or evolution in the optimal individual.
    count_no_evolution = 0
    # Initial Optimal Individual
    prev_optimal_ind = None

    ### Main Algorithm - Random Method
    for gen in range(n_gen): # from 0 to n_gen-1
        print("="*300)
        print("Gen:", gen)
        new_pop = toolbox.population(pop_size) # Generate new population with size equal to the given initial pop_size

        # Evaluate the new_pop
        with concurrent.futures.ProcessPoolExecutor(max_workers=concurrent_exec_params["n_processors"]) as executor:
            sub_new_pop = [ind for ind in executor.map(toolbox.eval_modifier, new_pop, chunksize=chunk_size, timeout=concurrent_exec_params["timeout"])]
            executor.shutdown()
        sub_new_pop = [ind for ind in sub_new_pop if is_fitness_tuple_valid(ind.fitness.values)]
        sub_new_pop_size = len(sub_new_pop)
        assert sub_new_pop_size <= pop_size
        print("sub_new_pop size:", sub_new_pop_size)
        print("sub_new_pop (evaluated) for next generation")
        print("\n")
        for _ in sub_new_pop:
            print(_.fitness.values,"|",_)
        print("-"*100)

        # Append the sub_new_pop valid individuals to subpop.
        # Note that this may not equal to the original population size, thus update pop_size after appending.
        subpop = subpop + sub_new_pop
        subpop_size = len(subpop) # This may not be equal to the initial pop_size.

        # Rank the top <pop_size> based on optimization direction. Note that the tail of list is the 'optimal' individual.
        if is_single_fitness: # One Fitness Value
            if optim_direction == "max":
                subpop = sorted(subpop, key=lambda ind: ind.fitness.values[0], reverse=False)
            if optim_direction == "min":
                subpop = sorted(subpop, key=lambda ind: ind.fitness.values[0], reverse=True)
        else: # Multi Fitness Values
            if optim_direction == "max":
                subpop = sorted(subpop,
                                key=lambda ind: transform_to_single_objective(ind.fitness.values, perf_stats.weights, method=single_fitness_method, fitness_idx=fitness_idx),
                                reverse=False)
            if optim_direction == "min":
                subpop = sorted(subpop,
                                key=lambda ind: transform_to_single_objective(ind.fitness.values, perf_stats.weights, method=single_fitness_method, fitness_idx=fitness_idx),
                                reverse=True)
        subpop = subpop[-pop_size:] # Only get the top N

        try:
            prev_optimal_ind = subpop[-1]  # Save the current optimal individual at gen=0 for comparing with optimal of next generation
        except IndexError: # We havnt found optimal individual because we havn't generated a valid population.
            prev_optimal_ind = None
            count_no_evolution += 1
        if prev_optimal_ind is not None:
            # Check if our current optimal is the same as the previous generation. count_no_evolution & "no_evol_after_ngen"
            if subpop[-1].fitness.values == prev_optimal_ind.fitness.values: # No Evolution
                print(f"There is no evolution at gen={gen}")
                if count_no_evolution > gp_algo_exit_params["no_evol_after_ngen"]: # Exit Criterion for No Evolution
                    print(f"Exit | We have not not evolved after {count_no_evolution} generations.")
                    return subpop
                else:
                    # Update the count
                    count_no_evolution += 1
                    prev_optimal_ind = subpop[-1]
            else: # Evolved/Improvement
                # Reset count_no_evolution
                count_no_evolution = 0
                # Update the new prev_optimal_ind
                prev_optimal_ind = subpop[-1]

        print("Population (evaluated) for next generation")
        print("\n")
        for _ in subpop:
            print(_.fitness.values,"|",_)
        print("\n")
        print("\n")
        try:
            print(f"Optimal Individual at Gen={gen} with fitness={subpop[-1].fitness.values} is {subpop[-1]}")
        except:
            pass

        # Update count_no_valid_pop and Check if we havn't produced any valid population after certain generations/iterations.
        if subpop_size == 0:
            if count_no_valid_pop > gp_algo_exit_params["no_valid_pop_after_n_gen"]:
                # Exit the loop when we generate no valid population.
                print(f"Exit | We have not generated a valid population after {count_no_valid_pop} generations.")
                return subpop
            count_no_valid_pop += 1 # Update count_no_valid_pop
        else:
            count_no_valid_pop = 0  # When we generate valid pop (i.e. we have at least 1 valid individual), then reset the counting.

        # Check if any of the evaluated individuals, if any, have satisfied all fitness targets criterion for exit.
        if gp_algo_exit_params["check_fitness_targets"]:
            try:
                if any((ind.fitness_achieved for ind in subpop)):
                    print("Exit | We have achieved fitness targets")
                    return subpop
            except Exception:
                raise Exception("We might have forgotten to set the fitness_targets in PerfStats Object.")

        # Check exit criterions.
        current_time = time.time()
        elapsed_time = current_time - start_time
        print()
        if gen > n_gen or elapsed_time >= maximum_duration:
            print(f"Exit | We have finished {n_gen} generations or we are passed the max duration {maximum_duration}")
            return subpop

    # return subpop # List[gp.PrimitiveTree] evaluated (i.e. with valid fitness property) gp.PrimitiveTree

# @_save_result_to_pickle("D:\Projects\FinDashAnalytics\Data\evoquant\evoquant standard gp strategies")
# @_measure_execution_time
def gp_main_algo_standard_gp(toolbox:base.Toolbox, optim_direction:str, perf_stats:PerfStats, in_population:List[gp.PrimitiveTree]=[],
                             pop_size:int=500, n_children=500,  n_gen=25, mut_pb=0.2, cx_pb=0.8,
                             single_fitness_method:str="weighted_sum", fitness_idx=None,
                             gp_algo_exit_params=dict(no_evol_after_ngen=5, check_fitness_targets=True, after_n_hours=24, no_valid_pop_after_n_gen=3),
                             concurrent_exec_params=dict(use_concurrent=True, n_processors=None, chunksize=None, timeout=None, allocate_memory_mb=None)
                             ) -> Union[List[gp.PrimitiveTree], List[None]]:
    from deap import tools
    import textwrap
    from deap.algorithms import varAnd, varOr
    import concurrent.futures
    import multiprocessing
    import math

    if concurrent_exec_params["n_processors"] is None: # Check Number of Logical Processors
        concurrent_exec_params["n_processors"] = multiprocessing.cpu_count()
    if isinstance(concurrent_exec_params["n_processors"], int) and concurrent_exec_params["n_processors"] > multiprocessing.cpu_count():
        raise ValueError("n_processors cannot be more than the cpu's logical processors.")

    if concurrent_exec_params["chunksize"] is None: # Check Chunk Size
        concurrent_exec_params["chunksize"] = math.floor(pop_size/concurrent_exec_params["n_processors"])
    assert isinstance(concurrent_exec_params["chunksize"], int) and concurrent_exec_params["chunksize"] < pop_size

    chunk_size = concurrent_exec_params["chunksize"]

    # Check if the given perf_stats and optim_direction are valid.
    if len(perf_stats.weights) == 1:
        is_single_fitness = True  # Set is_single_fitness. To be used later.
        if perf_stats.weights[0] > 0:
            optim_direction = "max"
        if perf_stats.weights[0] < 0:
            optim_direction = "min"
    else:  # Using Multi-Objective/Fitness Functions
        is_single_fitness = False  # Set is_single_fitness. To be used later.
        # Check if objective are maximization or minimization based on weights.
        if all([(w > 0) for w in perf_stats.weights]):
            optim_direction = "max"
        if all([(w < 0) for w in perf_stats.weights]):
            optim_direction = "min"
        else:  # Mix Direction. Which would also depend on the input of the user.
            if optim_direction is None or optim_direction not in ['max', 'min']:  # Check if optim_direction is given.
                raise ValueError("Optimization Direction must be given. Either 'max' or 'min'.")

    # Create Statistics and  MultiStatistics from tools
    temp_dict = {}
    for k in range(len(perf_stats.fitness_names)):
        raw_code = rf"""
            stats_fitness_{k} = tools.Statistics(key=lambda ind: np.array(ind.fitness.values[{k}])) # Individual's kth fitness values
            temp_dict["fitness_{k}"] = stats_fitness_{k}
            """
        exec(textwrap.dedent(raw_code))
    exec(r"""temp_dict["height"] = tools.Statistics(key=lambda ind: np.array(ind.height))""")
    exec(r"""temp_dict["length"] = tools.Statistics(key=lambda ind: np.array(len(ind)))""")
    mstats = tools.MultiStatistics(**temp_dict)  # Plug the keyword argument for each Statistics Object to MultiStatistics instantiation.
    for k in range(len(perf_stats.fitness_names)):
        raw_code = rf"""
            mstats.register("avg", np.nanmean, axis=0)  # Only calculate the non np.nan values. If all are np.nan, return np.nan
            mstats.register("median", np.nanmedian, axis=0)
            mstats.register("max", np.nanmax, axis=0)
            mstats.register("min", np.nanmin, axis=0)
            """
        exec(textwrap.dedent(raw_code))

    # Create Logbook from tools
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

    # Create an Executor instance
    use_concurrent = concurrent_exec_params["use_concurrent"]
    if use_concurrent:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=concurrent_exec_params["n_processors"])
        toolbox.register("map", executor.map, timeout=None, chunksize=chunk_size) # Temporarily register the executor.map in toolbox
    else:
        toolbox.register("map", map) # Use the basic map

    # Start timing for after_n_hours exit criterion
    import time
    start_time = time.time()
    maximum_duration = gp_algo_exit_params["after_n_hours"] * 60 * 60  # after_n_hours hours in seconds

    if len(in_population) == 0:
        count_no_valid_pop = 0
        while True:
            try:
                # Create and Evaluate the initial population.
                population = toolbox.population(pop_size)
                population = list(toolbox.map(toolbox.eval_modifier, population)) # It would apply toolbox.evaluate and modify the individual's fitness.values property
                population = [ind for ind in population if is_fitness_tuple_valid(ind.fitness.values)] # There is a possibility that this list can be empty
            except AttributeError:
                count_no_valid_pop += 1
            if len(population) >= 2: # Exit this loop if we generate at least 2 valid individual
                break
            else:
                if count_no_valid_pop >= gp_algo_exit_params["no_valid_pop_after_n_gen"]:
                    print(f"Exit | We have not generated a valid initial population after {count_no_valid_pop} tries.")
                    if use_concurrent:
                        executor.shutdown()  # Shutdown executor
                        toolbox.unregister("map")  # Unregister the map
                    return []
                else:
                    count_no_valid_pop += 1
        in_optim_inds = [] # Set in_optim_inds to be an empty list
    else: # Run this code if we have given a List of gp.PrimitiveTree (subclass of creator.Individual)
        assert type(in_population)==list and all([issubclass(type(ind), gp.PrimitiveTree) for ind in in_population]), "The input population is invalid"
        # Check is if each individual in the input population all have fitness.values property.
        assert all([is_fitness_tuple_valid(ind.fitness.values) for ind in in_population]), "The input population must have been evaluated and have valid fitness values"
        population = copy.deepcopy(in_population)
        # The algorithm from previous iteration (for multi-island) might have exited because some of the individual achieved fitness targets. Collect all optimal individuals and
        # store them in a variable. Append them in populations before returning.
        try:
            in_optim_inds = [copy.deepcopy(ind) for ind in population if ind.fitness_achieved==True]
            for ind in in_optim_inds:
                population.remove(ind) # Remove the optimal individuals from population
        except: # There may be an error when fitness targets is not set in PerfStats object.
            print("We did not set fitness targets.")
            in_optim_inds = []
    in_optim_inds = in_optim_inds

    # Check if any of the evaluated individuals, if any, have satisfied all fitness targets criterion for exit.
    if gp_algo_exit_params["check_fitness_targets"]:
        try:
            if any((ind.fitness_achieved for ind in population)):
                print("Exit | We have achieved fitness targets")
                return in_optim_inds + population
        except Exception: # Sometimes the ind.fitness_achieved is not set (i.e. ind.fitness_achieved=None)
            raise Exception("We might have forgotten to set the fitness_targets in PerfStats Object.")

    # Record and Log Initial Population Statistics.
    record = mstats.compile(population) if mstats is not None else {}
    logbook.record(gen=0, **record) # At Generation 0

    # Begin the generational process
    for gen in range(1, n_gen + 1):
        count_no_valid_pop = 0  # Reset count_no_valid_pop
        while True:
            # Vary the population
            # offspring = varOr(population, toolbox, lambda_=n_children, cxpb=cx_pb, mutpb=mut_pb) # Only Crossover and Mutation OR reproduction
            offspring = evo_varor(population, toolbox, lambda_=n_children, cxpb=cx_pb, mutpb=mut_pb)  # Only Crossover and Mutation OR reproduction
            offspring = list(toolbox.map(toolbox.eval_modifier, offspring)) # It would apply toolbox.evaluate and modify the individual's fitness.values property
            offspring = [ind for ind in offspring if is_fitness_tuple_valid(ind.fitness.values)]  # There is a possibility that this list can be empty
            if len(offspring) >= 2:  # Exit this loop if we generate at least 2 valid individual
                break
            else:
                if count_no_valid_pop >= gp_algo_exit_params["no_valid_pop_after_n_gen"]:
                    print(f"We have not generated a valid offspring after {count_no_valid_pop} tries.")
                    if use_concurrent:
                        executor.shutdown()  # Shutdown executor
                        toolbox.unregister("map")  # Unregister the map
                    return in_optim_inds + population
                else:
                    count_no_valid_pop += 1

        # Select the next generation population
        if is_single_fitness: # One Fitness Value
            if optim_direction == "max":
                population = sorted(population + offspring, key=lambda ind: ind.fitness.values[0], reverse=False)
            if optim_direction == "min":
                population = sorted(population + offspring, key=lambda ind: ind.fitness.values[0], reverse=True)
        else: # Multi Fitness Values
            if optim_direction == "max":
                population = sorted(population + offspring,
                                    key=lambda ind: transform_to_single_objective(ind.fitness.values, perf_stats.weights,
                                                                                  method=single_fitness_method,
                                                                                  fitness_idx=fitness_idx),
                                    reverse=False)
            if optim_direction == "min":
                population = sorted(population + offspring,
                                    key=lambda ind: transform_to_single_objective(ind.fitness.values,
                                                                                  perf_stats.weights,
                                                                                  method=single_fitness_method,
                                                                                  fitness_idx=fitness_idx),
                                    reverse=True)
        population = population[-pop_size:] # Rank from least optimal to optimal.

        # Check if there is evolution
        if gen == 1:
            # This code expect a population with at least 1 valid individual with fitness values. Because we are counting if we have not generated a
            # valid offspring list after some counting, then it will return an empty list.
            prev_optimal_ind = population[-1] # Set initial optimal valid individual from gen=1
            count_no_evolution = 0 # Set count_no_evolution
        else:
            if population[-1].fitness.values == prev_optimal_ind.fitness.values: # No Evolution
                if count_no_evolution >= gp_algo_exit_params["no_evol_after_ngen"]:
                    print(f"Exit | We have not evolved after {count_no_evolution+1} generations")
                    break
                else:
                    count_no_evolution += 1 # Update
            else: # Evolved
                count_no_evolution = 0 # Reset

        # Update the statistics with the new population
        record = mstats.compile(population) if mstats is not None else {}
        logbook.record(gen=gen, **record)
        # print(logbook.stream)

        # Check if any of the evaluated individuals, if any, have satisfied all fitness targets criterion for exit.
        if gp_algo_exit_params["check_fitness_targets"]:
            try:
                if any((ind.fitness_achieved for ind in population)):
                    print("Exit | We have achieved fitness targets")
                    break
            except Exception:
                raise Exception("We might have forgotten to set the fitness_targets in PerfStats Object.")

        # Check exit criterions.
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= maximum_duration:
            print(f"Exit | We are passed the max duration {maximum_duration} hours")
            break

    # print("\n")
    # for ind in population[-10:]:
    #     print(f"{ind.fitness}","|",ind)

    if use_concurrent:
        executor.shutdown()  # Shutdown executor
        toolbox.unregister("map")  # Unregister the map

    return in_optim_inds + population

def _right_shift_list(lst:list, shift:int):
    shift %= len(lst)  # Adjust shift value if it exceeds the length of the list
    return lst[-shift:] + lst[:-shift]

def ring_migration(populations:List[List[gp.PrimitiveTree]], n_migrants:int):
    """Returns a Modified Populations using Ring Migration technique.
    This function would implement a migration technique such that the top <n_migrants> from one population will be inserted to the other deme in a circular manner.
    In the standard gp algorithm, we implemented a ranking approach where the Individual on the left side of the list is the least fit individual, and
    the Individual on the right side of the list is the most fit individual. This is agnostic to the optimization direction (min/max).

    There is a possibility where some or all the demes are empty lists.
    """
    assert all([type(deme)==list for deme in populations]), "The Populations must be a List of Lists"
    assert n_migrants > 0 and isinstance(n_migrants, int), "The Number of Migrants must be a positive integer"

    n_demes = len(populations)
    temp_copy_pops = copy.deepcopy(populations) # Need the deep copy because some migrant might be accidentally inserted to other demes unintentionally.

    for i, j in zip(list(range(n_demes)), _right_shift_list(list(range(n_demes)), 1)): # Get 2-tuple of indexes of demes; from deme i to deme j
        # Get the top <n_migrants> from demes[i]
        # Insert top <n_migrants> from demes[i] to demes[j]
        populations[j] = temp_copy_pops[i][-n_migrants:] + populations[j]


def _main_algo(in_pop:list, standard_gp_args=tuple(), ngen:int=1, standard_gp_kwargs=dict()):
    args = copy.deepcopy(standard_gp_args)
    kwargs = copy.deepcopy(standard_gp_kwargs)
    kwargs["in_population"] = in_pop # Modify the in_population
    kwargs["pop_size"] = standard_gp_kwargs["pop_size"] # Modify the pop_size
    kwargs["n_children"] = standard_gp_kwargs["n_children"] # Modify the n_children
    kwargs["n_gen"] = ngen
    kwargs["concurrent_exec_params"]["use_concurrent"] = False
    return gp_main_algo_standard_gp(*args, **kwargs)

# @_save_result_to_pickle(r"D:\Projects\FinDashAnalytics\Data\evoquant\evoquant multi island strategies")
def gp_main_algo_multi_islands(*standard_gp_args, n_demes:int, n_migrants:int, init_n_gen=4, migration_rate=3,
                               **standard_gp_kwargs
                               ) -> Union[List[gp.PrimitiveTree], List[None]]:
    import copy
    import concurrent.futures
    from functools import partial
    import multiprocessing

    assert isinstance(n_demes, int) and n_demes >= 2, "The Number of Demes must be an integer and at least 2"
    assert n_demes <= multiprocessing.cpu_count(), f"The Number of Demes cannot be more than the cpu logical processors in this computer: {multiprocessing.cpu_count()}"
    assert isinstance(n_migrants, int) and n_migrants >= 1, "The Number of Migrants must be an integer and at least 1"

    # We will need this object for creating and shutting down toolbox.map. We use a deepcopy because when we apply map, we dont want it to be used inside the standard gp algo.
    tbx = copy.deepcopy(standard_gp_args[0])
    pf = copy.deepcopy(standard_gp_args[2])

    standard_gp_kwargs["concurrent_exec_params"]["use_concurrent"] = False # Make sure to switch off the concurrent when referencing the standard gp algorithm.

    n_gen = standard_gp_kwargs["n_gen"] # Get the n_gen from standard_gp_kwargs
    n_children = standard_gp_kwargs["n_children"]
    deme_pop_size = standard_gp_kwargs["pop_size"]

    if standard_gp_kwargs["concurrent_exec_params"]["n_processors"] != None:
        max_workers = standard_gp_kwargs["concurrent_exec_params"]["n_processors"]
        assert max_workers <= multiprocessing.cpu_count(), "The Max Worker parameter must be less than or equal to the number of cpu logical processors"
    else:
        max_workers = multiprocessing.cpu_count()

    # INIT_N_GEN = 7 # How many iteration to generate the initial demes

    main_algo_init_gen = partial(_main_algo, standard_gp_args=standard_gp_args, ngen=init_n_gen, standard_gp_kwargs=standard_gp_kwargs)
    main_algo_n_gen = partial(_main_algo, standard_gp_args=standard_gp_args, ngen=1, standard_gp_kwargs=standard_gp_kwargs)

    # Initial 'demes' is a List of Empty Lists.
    demes = [[] for _ in range(n_demes)]

    # Generate initial (evaluated/valid) demes:List[Union[List[gp.PrimitiveTree], List[None]]]
    print("Generating Initial Valid Demes")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    tbx.register("map", executor.map, timeout=None, chunksize=1)
    demes = list(tbx.map(main_algo_init_gen, demes)) # List[List[gp.PrimitiveTree]]
    tbx.unregister("map")
    executor.shutdown()

    print("We successfully finished generating the initial demes!")
    assert len(demes) == n_demes, "The Number of Demes Generated is not equal to the n_demes parameter"

    # Run the Main GP Loop
    for gen in range(1, n_gen+1):
        print("="*200)
        print(f"Generation {gen}")
        if gen % migration_rate == 0: # If the current generation is divisible by <migration_rate>, then perform migration
            print(f"We are performing migration at iteration {gen}")
            ring_migration(demes, n_migrants)

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        tbx.register("map", executor.map, timeout=None, chunksize=1)
        demes = list(tbx.map(main_algo_n_gen, demes))  # List[List[gp.PrimitiveTree]]
        tbx.unregister("map")
        executor.shutdown()

    print("\n")
    print("We have successfully finished the main loop!")
    out_population = []
    for deme in demes:
        for ind in deme:
            print(ind.fitness.values, "|", ind)
            out_population.append(ind) # There is no sorting/ranking

    w = standard_gp_args[2].weights
    single_fitness_method = standard_gp_kwargs["single_fitness_method"]
    single_fitness_idx = standard_gp_kwargs["fitness_idx"]
    if standard_gp_args[1] == "max":
        if len(w) == 1:
            return sorted(out_population, key=lambda ind: ind.fitness.values[0], reverse=False)
        else:
            return sorted(out_population,
                          key=lambda ind: transform_to_single_objective(ind.fitness.values, w, method=single_fitness_method, fitness_idx=single_fitness_idx),
                          reverse=False)
    if standard_gp_args[1] == "min":
        if len(w) == 1:
            return sorted(out_population, key=lambda ind: ind.fitness.values[0], reverse=True)
        else:
            return sorted(out_population,
                          key=lambda ind: transform_to_single_objective(ind.fitness.values, w,
                                                                        method=single_fitness_method,
                                                                        fitness_idx=single_fitness_idx),
                          reverse=True)

