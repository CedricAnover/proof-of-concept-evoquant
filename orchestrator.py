

"""
class _Strategy:
    SeriesBool
    PrimitiveTree
    Orchestrator

Or we can save all in a json file for the whole population.
"""

class Orchestrator:
    """
    Each user have different requirements for Indicators and Signals.

    Parameter
    - Symbol : Str (Required)
        The name of the symbol/ticker for the Dataset (e.g. "SPY"). This will be used as a tag.
    - Timeframe : Str (Required) {"1d", "1h", etc.}
        The granularity of the dataset. This will be used as a tag.
    - data : pd.DataDrame (Required)
        Pandas DataFrame that contain columns Date, Open, High, Low, Close, and Volume. Date column can be date or datetime. Volume is optional.
        Volume will be converted into a float.

    Properties
    - _raw_data : pd.DataDrame
    - df_dohlc[v] : pd.DataDrame
        This dataframe considers 'Date' as a column and not an index.
    - df_ohlc[v] : pd.DataDrame
        This dataframe considers 'Date' as an index.
    - terminal_types : List[Any]
        Contains the List of all terminal types. This is automatically modified when adding new Ephemeral Constants, and Primitives, that involve custom types.
        e.g. "if Quantile not in self.terminal_types: self.terminal_types.append(Quantile)"
    - input_gp : Tuple[SeriesBases]
        The input tuple for the main Primitive Set (pset). This is created once the Class is instantiated from the given dataset.
    ----------------------------------------------------
    - pset : gp.PrimitiveSetTyped
        DEAP gp.PrimitiveSetTyped. This will be used to add Ephemeral Constants and Primitives.
    - pset_mapping : Dict[Str -> Any]
        Dictionary containing all relevant functions and classes. This has to be modified with inspect and imported classes/functions from other evoquant modules.
    - toolbox : Toolbox
        DEAP Toolbox
    - perf_stats : PerfStats
    - strat_params : Dict
    - evo_bt_params : Dict
    - bt : backtesting.Backtest
    - translator : CTraderTranslator | MT4Translator | MT5Translator | TSTranslator | NJTranslator
    ----------------------------------------------------
    - final_pop : List[gp.PrimitiveTree|Individual] | List[None]
        List of Individuals/gp.PrimitiveTree when a Main GP Algorithm is finished running. This can be an empty list when
        the algorithm didn't find any optimal trading strategies. This could be due to Layer 2 filters.
    -
    -
    -
    -

    Methods
    - run(inf_loop=False)
    - get_optimal_strategy() -> gp.PrimitiveTree
    - set_orchestrator_instance_settings()
    -
    -
    -
    -
    -


    Usage:

    """
    def __new__(cls, *args, **kwargs):

        # Register the primitives in CTraderTranslator.register_primitive

        # Store EvoStrategy Configurations CTraderTranslator.store_strat_config(strat_params=<Strategy Config Dict>)
        return

    def __init__(self):
        pass


class Evolver:
    """This is one solution to implement the Main Genetic-Programming Algorithm and integrates the Backtesting Engine and GP Engine.

    Parameters:
    df_dohlcv : pd.DataFrame (Required)

    Properties:
    df_ohlcv : pd.DataFrame
        Processed data from df_dohlcv.
    main_input : Tuple[SeriesBase]
    terminal_types : List[Any|SeriesBase|ParameterBase]
        Will be modified as we add main_input, and other Terminals and Ephemirals. 
    pset : gp.PrimitiveSetTyped
    pset_mapping : Dict[Str->Class|Function|Variable]
    perf_stats : PerStats
    toolbox : base.Toolbox
    strat_params : Dict
    evo_bt_params : Dict
    bt_params : Dict
    bt : Backtest
    f2_ls : List[Tuple[str, str, str, float, dict]]
    gp_params : Dict
        This is for storing the parameters of the DEAP GP. Specifically, these are parameters for toolbox:deap.base.Toolbox.
        optim_direction : str ['max', 'min']
            But also depends on the given fitness functions.
        tree_min_height : int
        tree_max_height : int
        selector_params : Dict
        mutator_params : Dict
        crossover_params : Dict

    gp_algo_params : Dict
        This is for storing the parameters and configurations of the main algorithm(s). Standard, Multi-Islands, Random. Some parameters include:
        logical_cpu_processors : int (Optional), automatically finds the number and use them all
        chunksize : int
        main_algo : str ['standard', 'islands', 'random']
        exit_criterion = dict(no_evolution_after=10, n_generations=100, fitness_exit=("Sharpe", "ge", 1.5), after_n_hours=10, no_valid_pop_after_n_gen=None|Int)
        inf_loop : bool (Optional) Default=False
        population_size : int
        n_islands : int

    Methods:

    Construction:

    Ideas:
    evolver = Evolver(df_dohlcv)
    //Transforms the df_dohlcv into df_ohlcv where 'Date' is treated as Date|Datetime Column. It also treats 'Volume' as float if its included.
    evolver.preprocess_data()
        Run and clean the df_dohlcv. This function may also set values for other class properties such as cls.is_there_volume:Bool

    //input_gp is processed inside this class.

    evolver.raw_str : Str = r""""""
        Contains the generated code. Initially, this is an empty raw string. When this is invoked using @property, it will process all the 
        generated 'code components' and 'sum' it up into one executable valid python deap code.
    evolver.exec_raw_str()
        This method should execute the generated raw string using exec() function.
    evolver.save_as_python(folder_dir:str, name:str)
        Save the generated 'valid' deap gp code into a python code. folder_dir mus be an absolute path and name must be valid and end with .py

    //Each of the following functions would be appended to a raw string, which is to be executed in the main function using exec()
    // Note that the order of adding a code is important!!!
    evolver.generate_gp_code("Period", _period_code_func, min_value=5, max_value=500, ..) where _period_code_func(min_value, max_value, ..) -> Parametrized/Formatted Raw String
    evolver.generate_gp_code("Lag", max_value=10) where _lag_code_func(max_value, ..)
    evolver.generate_gp_code("BBands", std_min=0.5, std_max=0.3) _bbands_code_func(std_min, std_max, ..)

    Notes:
    - The template string generators might get included here and create a specific utils module for deap gp template generators.
    """
    pass