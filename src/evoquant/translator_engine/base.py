import typing

from deap import gp
# from evoquant.base import SeriesBool, ParameterBase
from evoquant import SeriesBool, ParameterBase

from functools import partial
from string import Template
import re
import copy

from typing import List, Union
from collections import OrderedDict

def _save_result_to_pickle(directory, file_extension='txt'):
    import os, datetime, uuid
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            result = func(*args, **kwargs)

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Generate a unique filename based on timestamp and UUID
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = os.path.join(directory, fr"""strategy_{timestamp}_{unique_id}.{file_extension}""")

            # Save the result as a pickle file
            with open(filename, "w") as file:
                file.write(result)

            return result
        return wrapper_func
    return decorator_func

class PTTranslator(gp.PrimitiveTree):
    """This Class extracts the necessary information of a PrimitiveTree that represents the trading strategy.

    Parameters

    Properties:
    length
    primitive_ls
    signal_primitive_ls
    indicator_primitive_ls
    composite_signal_primitive_ls
    simple_signal_primitive_ls
    composite_indicator_primitive_ls
    simple_indicator_primitive_ls
    non_primitive_ls

    Methods:

    individual : Union[gp.PrimitiveTree, deap.gp.Individual]
    ptt = PTTranslator(individual)
    """

    def __init__(self, content):
        super().__init__(content)

        self.length = len(self)

        # self.primitive_ls = list(set([self.index(elem) for elem in self if isinstance(elem, gp.Primitive)])) # To remove duplicates in indexes
        self.primitive_ls = []
        for i in range(self.length):
            elem = self[i]
            if isinstance(elem, gp.Primitive):
                self.primitive_ls.append(i)

        self.signal_primitive_ls = [idx for idx in self.primitive_ls if self[idx].ret == SeriesBool]
        self.indicator_primitive_ls = [idx for idx in self.primitive_ls if self[idx].ret != SeriesBool]
        assert set(self.signal_primitive_ls) | set(self.indicator_primitive_ls) == set(self.primitive_ls)

        # self.composite_signal_primitive_ls = [idx for idx in self.signal_primitive_ls if all([type(arg) == SeriesBool for arg in self[idx].args])]  # Defined by Types of Args
        # self.simple_signal_primitive_ls = [idx for idx in self.signal_primitive_ls if all([type(arg) != SeriesBool for arg in self[idx].args])] # Defined by Types of Args
        self.composite_signal_primitive_ls = [idx for idx in self.signal_primitive_ls if all([arg_type == SeriesBool for arg_type in self[idx].args])]  # Defined by Types of Args
        self.simple_signal_primitive_ls = [idx for idx in self.signal_primitive_ls if all([arg_type != SeriesBool for arg_type in self[idx].args])]  # Defined by Types of Args

        self.composite_indicator_primitive_ls = [idx for idx in self.indicator_primitive_ls if gp.PrimitiveTree(self[self.searchSubtree(idx)]).height > 1]
        self.simple_indicator_primitive_ls = [idx for idx in self.indicator_primitive_ls if gp.PrimitiveTree(self[self.searchSubtree(idx)]).height == 1]
        assert set(self.simple_signal_primitive_ls + self.composite_signal_primitive_ls + self.simple_indicator_primitive_ls + self.composite_indicator_primitive_ls) == \
               set(self.primitive_ls)

        # self.non_primitive_ls = [self.index(elem) for elem in self if not isinstance(elem, gp.Primitive)]
        self.non_primitive_ls = []
        for i in range(self.length):
            if not isinstance(self[i], gp.Primitive):
                self.non_primitive_ls.append(i)
        # Warn: A terminal (e.g. Volume) can appear many times in different indicators. And they reference the same thing.

    def primitive_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.non_primitive_ls]
    def signal_primitive_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.signal_primitive_ls]
    def indicator_primitive_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.indicator_primitive_ls]
    def composite_signal_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.composite_signal_primitive_ls]
    def simple_signal_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.simple_signal_primitive_ls]
    def composite_indicator_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.composite_indicator_primitive_ls]
    def simple_indicator_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.simple_indicator_primitive_ls]
    def non_primitive_list(self):
        return [str(gp.PrimitiveTree(self[self.searchSubtree(idx)])) for idx in self.non_primitive_ls]

    # (Deprecated)
    def format_idx(self, in_idx:int, *args) -> Union[None, str]:
        # This method is only for Primitives. Terminals will be formatted in another class. Or directly using .value property or .value_mapper(trade_platform) method.
        if isinstance(self[in_idx], gp.Terminal):  # If the node of in_idx is a Terminal, return None
            return
        assert self[in_idx].arity == len(args), "Positional Arguments must be equal to the Arity of the Primitive"
        return self[in_idx].format(*(f"{arg}" for arg in args)) # Very important as we can use this to replace the input argument value format

    def get_idx_args(self, in_idx:int) -> Union[List[int], None, int]:
        """Returns a list of indexes that are argument to the given index in_idx.
        This is equivalent to returning the children (not grandchildren of Primitives) of a node.
        This would return an empty list if the given index is the index of a Terminal. Because all Terminals have no children.
        """
        if isinstance(self[in_idx], gp.Terminal):  # If the node of in_idx is a Terminal, return None
            return []

        subtree = gp.PrimitiveTree(self[self.searchSubtree(in_idx)])
        # subtree_len = len(subtree)

        sub_sub_trees = self.get_idx_subtrees(in_idx)

        # Exclude a list(s) in sub_sub_trees that are sub-list of another list in sub_sub_trees.
        # The remaining roots sub-lists are the indexes of the argument(s) of in_idx
        out_lst = [lst[0] for lst in self._remove_sublists(sub_sub_trees)]  # Extract the roots
        return out_lst

    def get_idx_parent(self, in_idx:int) -> Union[int, None]:
        """Returns the index of the Parent node. This method will return None if the given index is the Root node.
        """
        assert in_idx < self.length
        if in_idx == 0: # in_idx is the Root (i.e. No Parent)
            return

        # root_sublists = self.get_idx_subtrees(0) # Get all the Tree Sublists of the root.
        parent_children = {i:self.get_idx_args(i) for i in range(self.length)} # Dict(Int -> Union[List[int], List[None]])

        for parent, children in parent_children.items():
            if in_idx in children:
                out_parent = parent
                assert isinstance(out_parent, int)
                return out_parent

    def get_idx_child_grandchild(self, in_idx:int, exclude_root:bool=True) -> Union[List[int], List[None]]:
        """Returns the List of Indexes of the Children & Grandchildren of given Node Index.
        This would return an empty list if the given node index is a Terminal node in a PrimitiveTree.
        It is optional to include the Parent/Grandparent in the List. By default, it's excluded.
        """
        if isinstance(self[in_idx], gp.Terminal):  # If the node of in_idx is a Terminal, return None
            return []
        tree = copy.deepcopy(self)
        subtree = gp.PrimitiveTree(self[self.searchSubtree(in_idx)])
        subtree_len = len(subtree)

        slice_start = self.searchSubtree(in_idx).start # Root
        slice_stop = self.searchSubtree(in_idx).stop # Last Element
        child_grandchild_idxs = [i for i in range(slice_start + 1, slice_stop)]  # Excluding the Start of the Slice.
        # child_grandchild_idxs = [self.index(subtree[i]) for i in range(1, subtree_len)]
        # child_grandchild_idxs = []
        # for i in range(1, subtree_len):
        #     elem = subtree[i]
        #     child_grandchild_idxs.append(i)

        if exclude_root:
            return child_grandchild_idxs
        else: # If we want the subtree root parent to be included in the start of the list.
            # return [in_idx] + child_grandchild_idxs
            return [slice_start] + child_grandchild_idxs

    def get_idx_subtrees(self, in_idx: int) -> Union[List[List[int]], List[None]]:
        """Returns the List of Subtrees (i.e. List of Indexes) of a given Node Index.
        This method returns an empty list if the given Node Index is a Terminal Node in a PrimitiveTree.
        """
        if isinstance(self[in_idx], gp.Terminal):  # If the node of in_idx is a Terminal, return None
            return []

        # This is the Slice of in_idx as a Subtree
        slice_start = self.searchSubtree(in_idx).start
        slice_stop = self.searchSubtree(in_idx).stop

        sub_sub_trees = []
        # Loop through the slice.
        # Has to be left-exclusive because we dont want to include the subtree itself as a subtree.
        for i in range(slice_start+1, slice_stop): # Left-Exclusive & Right-inclusive? i.e. exlude the root of index in_idx as a subtree of the main tree
            # Check if index i is a Terminal in main tree
            if isinstance(self[i], gp.Terminal):
                sub_sub_trees.append([i])
            else:
                sub_slice_start = self.searchSubtree(i).start
                sub_slice_stop = self.searchSubtree(i).stop
                temp_ls = list(range(sub_slice_start, sub_slice_stop)) # This one has to be left-inclusive
                sub_sub_trees.append(temp_ls)

        return sub_sub_trees

    def _remove_sublists(self, list_of_lists):
        # Create a set to store the indices of sublists to exclude
        excluded_indices = set()
        # Iterate over the list of lists
        for i in range(len(list_of_lists)):
            if i in excluded_indices:
                continue
            for j in range(i + 1, len(list_of_lists)):
                if j in excluded_indices:
                    continue
                sublist_a = list_of_lists[i]
                sublist_b = list_of_lists[j]
                # Check if sublist_a is a sublist of sublist_b
                if self._is_sublist(sublist_a, sublist_b):
                    excluded_indices.add(i)
                    break
                elif self._is_sublist(sublist_b, sublist_a):
                    excluded_indices.add(j)
        # Create a new list without the excluded sublists
        result = [lst for i, lst in enumerate(list_of_lists) if i not in excluded_indices]
        return result

    def _is_sublist(self, sublist_a, sublist_b):
        len_a = len(sublist_a)
        len_b = len(sublist_b)
        # If the length of sublist_a is greater than sublist_b, it cannot be a sublist
        if len_a > len_b:
            return False
        # Check if sublist_a is a sublist of sublist_b by comparing elements
        for i in range(len_b - len_a + 1):
            if sublist_b[i:i + len_a] == sublist_a:
                return True
        return False


class CTraderTranslator:
    """
        Example: Registering Primitive cTrader Expression Templates and Storing Strategy Configurations
        Resgister all the templates of primitives in DEAP and how they should be mapped to the target language (e.g. cTrader C#)
        Make sure to do this before making any instances of CTraderTranslator for a PrimitiveTree Instance

        e.g. CTraderTranslator.register_primitive("shift", r"Indicators.GetIndicator<Shift>({0}, {1}).Result") # (<Series>, <Lag>)

        Make sure to store the Strategy Configurations and Settings in the CTraderTranslator class before making any instances

        e.g. CTraderTranslator.store_strat_config(<Strategy Parameter Dictionary>)

        The above has to be set in the main evolution program in order to store the whole settings in memory before running the evolutionary algorithm.

        Example: Instantiating and Translating (for a PrimitiveTree instance).
        ptt = PTTranslator(<Some Strategy PrimitiveTree Instance>)
        ct_translator = CTraderTranslator(ptt)
        ct_translator.run_translation()

        Example: Generating a Valid Full Code for cTrader C#
        ct_translator.get_complete_code(filename="bot_code.txt", directory=r"<Directory>")
        """
    # The complete list of expected parameters (keyword args) for cTrader Complete Code. All the kwargs will be enclosed in $ or ${param_name} in string.Template
    _EVO_STRAT_PARAMS = \
    [
        "strategy_name",
        "direction",
        "trade_size",
        "exit_encoded_entry",
        "exit_after_n_bars",
        "exit_after_n_days",
        "exit_end_of_week",
        "exit_end_of_month",
        "exit_when_pnl_lessthan",
        "stop_loss",
        "take_profit",
        "enable_tsl"
    ]

    _strat_config: dict = dict()  # Strategy Configuration/Settings/Parameters of EvoStrategy(SignalStrategy) class. Needs to be modified in-memory before creating any instances.

    _trade_platform = "ctrader"
    _ct_version = "4.6.7"

    _primitive_to_ctrader: dict = {}  # Maps the DEAP Name of the Primitives to Valid 'Expressions' in cTrader. Dict(Str -> Callable)

    def __init__(self, ptt:PTTranslator):
        #-- All the Properties here are Getting Modified by a ptt instance. The only propery that should not be mutable is  _primitive_to_ctrader & _trade_platform
        self.ptt = ptt # PTTranslator instance.

        self._indicator_signal_code = ""  # Code Block for Valid cTrader Indicators and Signals. Stored in a Variable.

        self._idx_ctrader: dict = {}  # Maps PrimitiveTree index to valid syntax in CTRADER. This is for all PrimitiveTree Indexes, including the root signal
        self._idx_terminals: dict = {}  # Maps Terminal index to valid syntax in CTRADER
        self._idx_simple_indicators: dict = {}  # Maps Simple Indicator Primitives index to valid syntax in CTRADER
        self._idx_composite_indicators: dict = {}  # Maps Composite Indicator Primitives index to valid syntax in CTRADER
        self._idx_simple_signals: dict = {}  # Maps Simple Signal Primitives index to valid syntax in CTRADER
        self._idx_composite_signals: dict = {}  # Maps Composite Signal Primitives index to valid syntax in CTRADER

        self._root_signal:str = ""  # The Final Valid CTRADER Syntax for Root Signal

        # Run the run_translation & generate_indicator_signal_code methods
        self.run_translation()
        self.generate_indicator_signal_code()

    @classmethod
    def translate_terminal(cls, obj:gp.Terminal): # DEAP -> CTRADER
        """Return a Valid cTrader # Syntax for  Non-Primitive or Terminal Object from DEAP"""
        assert isinstance(obj, gp.Terminal), "The input must be a gp.Terminal object"

        def _contains_word_ignore_case(string, word): return word.lower() in string.lower()

        _value = obj.value # Extract the value of the gp.Terminal object (Not a ParameterBase!!!)

        if isinstance(_value, str): # This could be a SeriesFloat/SeriesPrice/SeriesOpen/.../SeriesVolume/SeriesDate
            # Make sure to Clean
            if _contains_word_ignore_case(_value, "date"): return r"Date"
            if _contains_word_ignore_case(_value, "open"): return r"Open"
            if _contains_word_ignore_case(_value, "high"): return r"High"
            if _contains_word_ignore_case(_value, "low"): return r"Low"
            if _contains_word_ignore_case(_value, "close"): return r"Close"
            if _contains_word_ignore_case(_value, "volume"): return r"Volume"
        else:
            assert issubclass(type(_value), ParameterBase), "The input object's value type must be a ParameterBase"
            if isinstance(_value.value, (int, float)):
                return _value.value_mapper(cls._trade_platform) # Sometimes an Int is mapped to a string such as DayOfWeek
            else: # Very likely to be a string. Must be matched with a valid object in C# cTrader (Enums or Interfaces)
                assert isinstance(_value.value, str), "The value is neither a string nor numeric!"
                return _value.value_mapper(cls._trade_platform)

    @property
    def primitive_to_ctrader(self) -> callable: # DEAP -> CTRADER
        return self._primitive_to_ctrader

    @property
    def idx_ctrader(self):
        return self._idx_ctrader

    @property
    def root_signal(self):
        return self._root_signal

    @property
    def indicator_signal_code(self):
        return self._indicator_signal_code

    @classmethod
    def register_primitive(cls, name:str, template:str):
        """
        Example Usage:
        translator = CTraderTranslator(...)
        translator.register_primitive("rsi", r"Indicators.RelativeStrengthIndex({0}, {1}).Result") # Use of Positional Arguments
        translator.primitive_to_ctrader["rsi"]("<ctrader Series>", CTraderTranslator.translate_terminal(<Terminal Object>))
        Need to make sure that the order and number of arguments must be exact!
        """
        cls._primitive_to_ctrader[name] = lambda *args: template.format(*args)

    # TODO (In-Progress) Implement a class method for checking, cleaning, and storing strategy params/config/settings for full code.
    @classmethod
    def store_strat_config(cls, strat_params:dict):
        """
        strat_params : Dict
            This dictionary should contain all the parameters for EvoStrategy(SignalStrategy) class.

        This method will be used to store the strategy parameters. This is important for the full code in ctrader.
        This will be supported by a private method that check and clean the strategy parameters/settings.

        Class Property _strat_config will be modified and stored in the Class, other instances can access it.

        Idea:
        Each parameter in EvoStrategy must have their own implementation on how to convert the EvoStrategy DEAP parameters to
        a valid ctrader parameters.
        """
        # Goal: Modify the cls._strat_config dictionary plus some other required parameters outside EvoStrategy
        assert len(strat_params.keys()) == len(cls._EVO_STRAT_PARAMS), "The given strategy parameter dictionary does not match the size of _EVO_STRAT_PARAMS"
        assert all([(k in cls._EVO_STRAT_PARAMS) for k in strat_params.keys()]), "There is some element in the Strategy Params that does not match the keys in _EVO_STRAT_PARAMS"

        for k in cls._EVO_STRAT_PARAMS:
            if k == "strategy_name":
                cls._strat_config[k] = r'"{0}"'.format(strat_params[k])
            if k == "direction":
                if strat_params[k] == "LongOnly":
                    cls._strat_config[k] = r'TradeDirection.LongOnly'
                if strat_params[k] == "ShortOnly":
                    cls._strat_config[k] = r'TradeDirection.ShortOnly'
                if strat_params[k] == "LongShort":
                    cls._strat_config[k] = r'TradeDirection.LongShort'
            if k == "trade_size":
                if isinstance(strat_params[k], int) and strat_params[k] >= 1:
                    # Check if its at least the minimum volume amount in ctrader.
                    cls._strat_config[k] = r"{0}".format(strat_params[k])
                elif isinstance(strat_params[k], float) and strat_params[k] > 0 and strat_params[k] < 1:
                    # Warn: in cTrader this will be dynamic depending on the state of the Account's Equity and Margin.
                    cls._strat_config[k] = r"{0}".format(strat_params[k])
                else:
                    print("The trade size must be a valid numeric - Volume in Units or Fraction of Equity. Defaulting to 99% of Equity")
                    cls._strat_config[k] = r"{0}".format(0.99) # Use Default 99% of Balance
            if k == "exit_encoded_entry":
                if strat_params[k]: # if True
                    cls._strat_config[k] = r"true"
                else: # if False
                    cls._strat_config[k] = r"false"
            if k == "exit_after_n_bars":
                if strat_params[k] is None:
                    cls._strat_config[k] = r"null"
                else:
                    assert isinstance(strat_params[k], int) and strat_params[k] >= 1, "exit_after_n_bars must be a valid positive integer"
                    cls._strat_config[k] = r"{}".format(strat_params[k])
            if k == "exit_after_n_days":
                if strat_params[k] is None:
                    cls._strat_config[k] = r"null"
                else:
                    assert isinstance(strat_params[k], int) and strat_params[k] >= 1, "exit_after_n_days must be a valid positive integer"
                    cls._strat_config[k] = r"{}".format(strat_params[k])
            # TODO: Implement mapping for exit_end_of_week
            if k == "exit_end_of_week":
                if strat_params[k]:
                    cls._strat_config[k] = r"true"
                else:
                    cls._strat_config[k] = r"false"
            # TODO: Implement mapping for exit_end_of_month
            if k == "exit_end_of_month":
                if strat_params[k]:
                    cls._strat_config[k] = r"true"
                else:
                    cls._strat_config[k] = r"false"
            if k == "exit_when_pnl_lessthan":
                if strat_params[k] is None:
                    cls._strat_config[k] = r"null"
                else:
                    assert isinstance(strat_params[k], (float, int)) and strat_params[k] < 0, "exit_when_pnl_lessthan must be a valid negative number"
                    cls._strat_config[k] = r"{}".format(strat_params[k])
            if k == "stop_loss":
                if len(strat_params[k]) == 0:
                    cls._strat_config[k] = r"null"
                elif strat_params[k][1] == "Percent":
                    # Convert to valid decimal
                    cls._strat_config[k] = r"({0}*Open.LastValue)/Symbol.PipSize".format(strat_params[k][0]/100)
                elif strat_params[k][1] == "Pip":
                    assert strat_params[k][0] > 0, "Pips represents a price range, this cannot be negative in evoquant context"
                    cls._strat_config[k] = r"{0}".format(strat_params[k][0]) # Already in Valid Pips
                elif strat_params[k][1] == "Point":
                    # TODO: Implement mapping for Fixed Stop-Loss 'Point'
                    raise
                else:
                    cls._strat_config[k] = r"null"
            if k == "take_profit":
                if len(strat_params[k]) == 0:
                    cls._strat_config[k] = r"null"
                elif strat_params[k][1] == "Percent":
                    # Convert to valid decimal
                    cls._strat_config[k] = r"({0}*Open.LastValue)/Symbol.PipSize".format(strat_params[k][0] / 100)
                elif strat_params[k][1] == "Pip":
                    assert strat_params[k][0] > 0, "Pips represents a price range, this cannot be negative in evoquant context"
                    cls._strat_config[k] = r"{0}".format(strat_params[k][0])
                elif strat_params[k][1] == "Point":
                    # TODO: Implement mapping for Fixed Take-Profit 'Point'
                    raise
                else:
                    cls._strat_config[k] = r"null"
            if k == "enable_tsl":
                if strat_params[k]:
                    cls._strat_config[k] = r"true"
                else:
                    cls._strat_config[k] = r"false"

    def run_translation(self):
        """This method will process and translate the deap string results to a valid ctrader syntax
        """
        # Process the Terminals. Cannot be an empty list.
        assert len(self.ptt.non_primitive_ls) != 0, "List of Terminals Cannot be Empty!"
        for idx in self.ptt.non_primitive_ls:
            self._idx_terminals[idx] = self.translate_terminal(self.ptt[idx])
            self._idx_ctrader[idx] = self._idx_terminals[idx] # All Indexes to CTRADER

        # print(f"Done Processing Terminals!")
        # print("idx_ctrader: ", self._idx_ctrader)
        # for k, v in self._idx_terminals.items():
        #     print("(", k, ",", v, ")")
        # print()

        # Process the Simple Indicators
        if len(self.ptt.simple_indicator_primitive_ls) != 0:
            for idx in self.ptt.simple_indicator_primitive_ls:
                args_idxs = self.ptt.get_idx_args(idx) # List[int] and cannot be empty. Assumed to be Terminals
                assert len(args_idxs) > 0, "List of Arguments for Primitives cannot be Empty!"
                assert all([isinstance(self.ptt[i], gp.Terminal) for i in args_idxs]), "Arguments for a Simple Indicator must be Terminals!"
                prim_name = self.ptt[idx].name # Get the Deap name of the primitive
                temp_arg_map = tuple(map(lambda i: self._idx_ctrader[i], args_idxs))
                self._idx_simple_indicators[idx] = self.primitive_to_ctrader[prim_name](*temp_arg_map)
                self._idx_ctrader[idx] = self._idx_simple_indicators[idx] # All Indexes to CTRADER

        # print(f"Done Processing Simple Indicators!")
        # print("idx_ctrader: ", self._idx_ctrader)
        # for k,v in self._idx_simple_indicators.items():
        #     print("(", f"idx={k}", ",", v, ")")
        # print()

        # Process the Composite Indicators (self._idx_composite_indicators)
        # Note: Some composite indicators are composed of other composite indicators and simple indicator
        if len(self.ptt.composite_indicator_primitive_ls) != 0:
            # We need to modify and translate composite indicators with the least length first before the longest length.
            temp_ls = [self.ptt.get_idx_child_grandchild(i, exclude_root=False) for i in self.ptt.composite_indicator_primitive_ls] # List[List[int]]
            temp_ls = sorted(temp_ls, key=len, reverse=False) # Sort a List of List of composite indicators (indexes) from Shortest to Longest Length
            temp_dict = {ls[0]:ls[1:] for ls in temp_ls}
            for idx, child_grandchild in temp_dict.items():
                assert idx not in self._idx_composite_indicators.keys(), "This cant happen as we are starting the loop from least length composite indicator"
                assert all([i in self._idx_ctrader.keys() for i in child_grandchild]), f"Some child or grandchild of Composite Indicator {idx} is not processed"
                    # continue # This cant happen as we are starting from the Composite Indicator with least length
                args_idxs = self.ptt.get_idx_args(idx) # List of Children (Argument)
                # print(f"Deap of idx={idx}: ", str(self.ptt[self.ptt.searchSubtree(idx)]))
                # parent_idx = self.ptt.get_idx_parent(idx) # Index of the Parent
                prim_name = self.ptt[idx].name # Get the Primitive Name
                # Note: Children of Composite Indicators can be Primitive and Terminal. We already processed the terminal. So check if the Primitive Children is processed.
                # Otherwise skip. Remember that Primitive Children can be Simple or Composite Indicator, but Simple Indicators is already Processed.
                # Need to reference _idx_ctrader because every non-composite indicators and primitives have been processed
                temp_arg_map = tuple(map(lambda i: self._idx_ctrader[i], args_idxs))
                self._idx_composite_indicators[idx] = self.primitive_to_ctrader[prim_name](*temp_arg_map)
                self._idx_ctrader[idx] = self._idx_composite_indicators[idx] # All Indexes to CTRADER

        # print(f"Done Processing Composite Indicators!")
        # print("idx_ctrader: ", self._idx_ctrader)
        # for k, v in self._idx_composite_indicators.items():
        #     print("(", f"{k}", ",", v, ")")
        # print()

        # Process Simple Signals
        # Note: Simple Signals is characterized by the argument type, which are non-SeriesBool. All arguments must already been processed.
        # Note that there is a possibility that the Root is a Simple Signal.
        # Remark: Sometimes if Composite Indicators is empty, there for it may throw an error in the following code block for Simple Signals.
        if len(self.ptt.simple_signal_primitive_ls) != 0:
            for idx in self.ptt.simple_signal_primitive_ls:
                assert idx not in self._idx_ctrader.keys(), f"It seems that Simple Signal idx={idx} has been processed and included in _idx_ctrader"
                assert idx not in self._idx_simple_signals.keys(), f"It seems that Simple Signal idx={idx} has been processed and included in _idx_simple_signals"
                args_idxs = self.ptt.get_idx_args(idx)  # List of Children (Argument)
                # parent_idx = self.ptt.get_idx_parent(idx)  # Index of the Parent
                prim_name = self.ptt[idx].name # Get the Primitive Name

                temp_arg_map = tuple(map(lambda i: self._idx_ctrader[i], args_idxs))
                self._idx_simple_signals[idx] = self.primitive_to_ctrader[prim_name](*temp_arg_map)
                self._idx_ctrader[idx] = self._idx_simple_signals[idx] # All Indexes to CTRADER

        # print(f"Done Processing Simple Signals!")
        # print("idx_ctrader: ", self._idx_ctrader)
        # for k, v in self._idx_simple_signals.items():
        #     print("(", f"{k}", ",", v, ")")
        # print()

        # Process Composite Signals
        if len(self.ptt.composite_signal_primitive_ls) != 0:
            # print("Processing Composite Signals")
            # print("Composite Signal Indexes:", self.ptt.composite_signal_primitive_ls)
            assert all([i not in self._idx_ctrader.keys() for i in self.ptt.composite_signal_primitive_ls]), "One of the Composite Signal Index has been processed before starting!"
            # Warn: Some Composite Signal may depend on other composite signal that has not been proccessed yet!
            #   We can solve this using reverse on the self.ptt.composite_signal_primitive_ls because the root will be the main composite signal.
            for idx in reversed(self.ptt.composite_signal_primitive_ls):
                # print(f"processing composite signal idx={idx} ...")
                # print(str(self.ptt[self.ptt.searchSubtree(idx)]))
                assert idx not in self._idx_ctrader.keys() and idx not in self._idx_composite_signals.keys(), f"It seems that Composite Signal idx={idx} has been processed."
                args_idxs = self.ptt.get_idx_args(idx)  # List of Children (Argument)
                assert all([(i in self.ptt.signal_primitive_ls) for i in args_idxs]), "One of the Argument Index is Not a Signal! Fix this!"
                assert all([i in self._idx_ctrader.keys() for i in args_idxs]), "One of the Argument Index is not processed and inserted in _idx_ctrader keys!"
                # print("Arg Indexes:", args_idxs)
                prim_name = self.ptt[idx].name  # Get the Primitive Name
                # print("Primitive Name:", prim_name)
                temp_arg_map = tuple(map(lambda i: self._idx_ctrader[i], args_idxs)) # Get the processed arguments
                self._idx_composite_signals[idx] = self.primitive_to_ctrader[prim_name](*temp_arg_map)
                self._idx_ctrader[idx] = self._idx_composite_signals[idx]  # All Indexes to CTRADER

        # print(f"Done Processing Composite Signals!")
        # print("idx_ctrader: ", self._idx_ctrader)
        # for k, v in self._idx_composite_signals.items():
        #     print("(", f"{k}", ",", v, ")")
        # print()

    def generate_indicator_signal_code(self) -> str:
        """This method would return a valid C# cTrader code block for the list of variables with values equal to the expressions in _idx_ctrader dictionary.

        cTrader C# Format
        var indicator<num> = <Expr|Value>;
        var signal<num> = <Expr|Value>;

        Example Output:
        var indicator0 = <Expr>;
        var indicator1 = <Expr>;
        ...
        var indicator4 = <Expr>;
        var indicator5 = <Expr>;
        ...
        var simple_signal1 = <Expr>;
        var simple_signal2 = <Expr>;
        ...
        var composite_signal1 = <Expr>;
        var composite_signal2 = <Expr>;
        ...
        var root_signal = <Expr>
        """

        def format_string(*args):
            # This would create a new line.
            # Example:
            # input_string = format_string(r"Hello", r"World", r"How", r"are", r"you?")
            # print(input_string)
            formatted_string = "\n".join(args)
            return formatted_string

        # Create Code for Simple Indicator Primitives
        i_d = 0 # For the indicator number e.g. indicator<num>
        simple_indicator_temp_dict = OrderedDict() # Dict(Str->Str) For the variable name and expression e.g. "indicator0":<cTrader Expr> then clean later as "var indicator0 = <cTrader Expr>;"
        simple_indicator_str_ls = [] # To be modified if not zero with [r"// Simple Indicators"]+simple_indicator_str_ls
        temp_mapper_simple_indicator = OrderedDict() # Dict(Str -> Dict(Str->Str)) This is to map the original ctrader code expression to the key-value pair of simple_indicator_temp_dict
        # Process Simple Indicators
        if len(self.ptt.simple_indicator_primitive_ls) != 0:
            for idx, ct_code_expr in self._idx_simple_indicators.items():
                simple_indicator_temp_dict[fr"indicator{i_d}"] = ct_code_expr # Variable String to CTRADER expression
                temp_mapper_simple_indicator[ct_code_expr] = {fr"indicator{i_d}": ct_code_expr} # Will be useful for signals.
                simple_indicator_str_ls.append(fr"var indicator{i_d} = {ct_code_expr};") # Append the 1 line code
                i_d += 1 # Update

        # print("Simple Indicators")
        # print(simple_indicator_str_ls)

        # Create Code for Composite Indicator Primitives
        composite_indicator_temp_dict = OrderedDict()
        composite_indicator_str_ls = []
        temp_mapper_composite_indicator = OrderedDict()  # OrderedDict(Str -> Dict(Str->Str)) This is to map the original ctrader code expression to the key-value pair of composite_indicator_temp_dict
        # Note: Need to iterate backwards in temp_mapper_composite_indicator, because we want to substitute variable of a more complex expression
        if len(self.ptt.composite_indicator_primitive_ls) != 0:
            temp_ls = [self.ptt.get_idx_child_grandchild(i, exclude_root=False) for i in self.ptt.composite_indicator_primitive_ls]  # List[List[int]]
            temp_ls = sorted(temp_ls, key=len, reverse=False)  # Sort a List of List of composite indicators (indexes) from Shortest to Longest Length
            temp_dict = {ls[0]: ls[1:] for ls in temp_ls}
            # for idx, ct_code_expr in self._idx_composite_indicators.items():
            # print("Testing Composite Indicator Primitives Code")
            for idx in list(temp_dict.keys()):
                ct_code_expr = self._idx_composite_indicators[idx] # Original
                # There is a possibility that a simple indicator or previous composite indicator in the loop is presset in current composite indicator.
                ct_code_expr_ = copy.deepcopy(ct_code_expr) # Just to make sure we dont modify ct_code_expr from dictionary _idx_composite_indicators
                # for variable_str, expr_str in composite_indicator_temp_dict.items(): # Check if some previous composite indicator expression is present in ct_code_expr
                #     if expr_str in ct_code_expr_:
                #         ct_code_expr_ = ct_code_expr_.replace(expr_str, variable_str)
                for original_expr, composite_indicator_dict in reversed(temp_mapper_composite_indicator.items()):
                    # Lets processed composite indicators is present in the current composite indicator
                    if original_expr in ct_code_expr:
                        var_str_repl = next(iter(composite_indicator_dict.keys()))
                        ct_code_expr_ = ct_code_expr_.replace(original_expr, var_str_repl) # Only One Key representing the variable
                for variable_str, expr_str in reversed(simple_indicator_temp_dict.items()): # Check if some simple indicator expression is present in ct_code_expr
                    if expr_str in ct_code_expr:
                        ct_code_expr_ = ct_code_expr_.replace(expr_str, variable_str)
                composite_indicator_temp_dict[fr"indicator{i_d}"] = ct_code_expr_  # Variable String to CTRADER expression
                temp_mapper_composite_indicator[ct_code_expr] = {fr"indicator{i_d}": ct_code_expr_}  # Will be useful for signals.
                composite_indicator_str_ls.append(fr"var indicator{i_d} = {ct_code_expr_};")  # Append the 1 line code
                i_d += 1  # Update

        # print("Composite Indicator")
        # print(composite_indicator_str_ls)

        i_d = 0 # Reset id for the Signals
        temp_root_signal = None # Initialize temp_root_signal, at the end of this method, the value should not be None

        # Create Code for Simple Signal Primitives
        simple_signal_temp_dict = OrderedDict()
        simple_signal_str_ls = []
        temp_mapper_simple_signal = OrderedDict()  # Dict(Str -> Dict(Str->Str)) This is to map the original ctrader code expression to the key-value pair of simple_temp_dict
        if len(self.ptt.simple_signal_primitive_ls) != 0:
            temp_ls = [self.ptt.get_idx_child_grandchild(i, exclude_root=False) for i in self.ptt.simple_signal_primitive_ls]  # List[List[int]]
            temp_ls = sorted(temp_ls, key=len, reverse=True)  # Sort a List of List of composite indicators (indexes) from Longest to Shortest Length
            temp_dict = {ls[0]: ls[1:] for ls in temp_ls}
            for idx in list(temp_dict.keys()):
                ct_code_expr = self._idx_simple_signals[idx]  # Original
                ct_code_expr_ = copy.deepcopy(ct_code_expr)  # Make a Deep Copy
            # for idx, ct_code_expr in self._idx_simple_signals.items():
                # Need to do Composite Indicators First Because they are more complex in terms of expressions
                for original_expr, composite_indicator_dict in reversed(temp_mapper_composite_indicator.items()): # Reverse the iteration on OrderedDict
                    if original_expr in ct_code_expr:
                        ct_code_expr_ = ct_code_expr_.replace(original_expr, next(iter(composite_indicator_dict.keys())))
                # Need to Check Simple Indicators because sometimes we are comparing Simples Indicators with some Complex Composite Indicators
                for variable_str, expr_str in reversed(simple_indicator_temp_dict.items()): # Reverse the iteration on OrderedDict
                    if expr_str in ct_code_expr:
                        ct_code_expr_ = ct_code_expr_.replace(expr_str, variable_str)
                if ct_code_expr == self._idx_ctrader[0]:  # Check if Current Simple Signal is the Root Signal. If it is, store the modified value in the property
                    self._root_signal = ct_code_expr_  # Store the modified value in the property _root_signal
                    temp_root_signal = fr"signal{i_d}" # Temporarily store the root_signal where its value depend on either simple signal or composite signal
                    # simple_signal_str_ls.append(fr"var root_signal = {ct_code_expr_};")
                simple_signal_temp_dict[fr"signal{i_d}"] = ct_code_expr_  # Variable String to CTRADER expression
                temp_mapper_simple_signal[ct_code_expr] = {fr"signal{i_d}": ct_code_expr_}
                simple_signal_str_ls.append(fr"var signal{i_d} = {ct_code_expr_};")  # Append the 1 line code

                # # If the Current ct_code_expr is the root signal, then just store the ctrader variable of the current ct_code_expr in root to avoid extra computation in ctrader
                # if ct_code_expr == self._idx_ctrader[0]:  # Check if Current Simple Signal is the Root Signal. If it is, store the modified value in the property
                #     self._root_signal = fr"signal{i_d}"  # Store the modified value in the property _root_signal

                i_d += 1  # Update

        # print("Simple Signals")
        # print(simple_signal_str_ls)

        # Create Code for Composite Signal Primitives
        composite_signal_temp_dict = OrderedDict()
        composite_signal_str_ls = []
        temp_mapper_composite_signal = OrderedDict()
        if len(self.ptt.composite_signal_primitive_ls) != 0:
            for idx, ct_code_expr in self._idx_composite_signals.items():
                ct_code_expr_ = copy.deepcopy(ct_code_expr)
                # Check if the processed Composite Signal is present in the current
                for original_expr, composite_signal_dict in reversed(temp_mapper_composite_signal.items()):
                    if original_expr in ct_code_expr:
                        ct_code_expr_ = ct_code_expr_.replace(original_expr, next(iter(composite_signal_dict.keys()))) # Replace the Original with a Variable from processed Composite Signals
                for original_expr, simple_signal_dict in reversed(temp_mapper_simple_signal.items()):
                    if original_expr in ct_code_expr:
                        ct_code_expr_ = ct_code_expr_.replace(original_expr, next(iter(simple_signal_dict.keys())))
                if ct_code_expr == self._idx_ctrader[0]: # Check if Current Composite is the Root Signal. If it is, store the modified value in the property
                    self._root_signal = ct_code_expr_ # Store the modified value in the property _root_signal
                    temp_root_signal = fr"signal{i_d}" # Temporarily store the root_signal where its value depend on either simple signal or composite signal
                composite_signal_temp_dict[fr"signal{i_d}"] = ct_code_expr_  # Variable String to CTRADER expression
                temp_mapper_composite_signal[ct_code_expr] = {fr"signal{i_d}": ct_code_expr_}
                composite_signal_str_ls.append(fr"var signal{i_d} = {ct_code_expr_};")  # Append the 1 line code

                # # If the Current ct_code_expr is the root signal, then just store the ctrader variable of the current ct_code_expr in root to avoid extra computation in ctrader
                # if ct_code_expr == self._idx_ctrader[0]: # Check if Current Composite is the Root Signal. If it is, store the modified value in the property
                #     self._root_signal = fr"signal{i_d}" # Store the modified value in the property _root_signal

                i_d += 1  # Update

        # print("Composite Signals")
        # print(composite_signal_str_ls)

        # x1 = format_string(*tuple(simple_indicator_str_ls))
        # x2 = format_string(*tuple(composite_indicator_str_ls))
        # x3 = format_string(*tuple(simple_signal_str_ls))
        # x4 = format_string(*tuple(composite_signal_str_ls))
        # out_str = format_string(*(r"// Simple Indicators", x1,
        #                           r"// Composite Indicators", x2,
        #                           r"// Simple Signals", x3,
        #                           r"// Composite Signals", x4,
        #                           r"// Root Signal", fr"var root_signal = {self._root_signal};"))
        # self._indicator_signal_code = out_str

        assert temp_root_signal is not None, "temp_root_signal has not changed!"

        x1 = format_string(*tuple([r"// Simple Indicators"] + simple_indicator_str_ls)) if len(simple_indicator_str_ls) != 0 else ""
        x2 = format_string(*tuple([r"// Composite Indicators"] + composite_indicator_str_ls)) if len(composite_indicator_str_ls) != 0 else ""
        x3 = format_string(*tuple([r"// Simple Signals"] + simple_signal_str_ls)) if len(simple_signal_str_ls) != 0 else ""
        x4 = format_string(*tuple([r"// Composite Signals"] + composite_signal_str_ls)) if len(composite_signal_str_ls) != 0 else ""
        out_str = format_string(*(x1, x2, x3, x4, r"// Root Signal", fr"var root_signal = {temp_root_signal};"))
        self._indicator_signal_code = out_str


    # TODO (Testing) Implement a class method to generate a full & valid ctrader code.
    # @_save_result_to_pickle(r"D:\Projects\FinDashAnalytics\Data\evoquant\Strategy cTrader Codes")
    def get_complete_code(self, directory:str=None) -> str:
        """Returns a Complete and Valid cTrader C# Code
        Options: Return it as a String or Store it in a txt file (not C#).
        If the user chose to store the code in a txt file, they will have to open, then copy-paste in ctrader platform.

        ct_translator.get_complete_code()
        """
        # TODO (Testing) Complete the Full Template in cTrader C# including indicators, signals, exits, and other parameters/configurations/settings.
        # Return Valid Editted cTrader Full Code for an Instance.
        global _CTRADER_FULL_TEMPLATE
        import textwrap

        template = Template(_CTRADER_FULL_TEMPLATE).safe_substitute(indicator_signal_code=self._indicator_signal_code, **self._strat_config)
        template = textwrap.dedent(template)

        # Store the Strategy Instance Full cTrader C# Code in the given directory
        if directory is not None:
            import datetime, uuid, os
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # Get Current Date Time
            unique_id = str(uuid.uuid4())[:8]   # Generate Random ID
            filename = os.path.join(directory, fr"ctrader_code_{timestamp}_{unique_id}.txt")
            # Save the result as a pickle file
            with open(filename, "w") as file:
                file.write(template)

        return template



# Idea here is to create a variable within translator_engine.base for full-code templates in ctrader, mt4, mt5, easylanguage, ninjatrader, etc.
_CTRADER_FULL_TEMPLATE = \
r"""
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using cAlgo.API;
using cAlgo.API.Collections;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    public enum TradeDirection
    {
        LongOnly,
        ShortOnly,
        LongShort
    }


    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class TemplateBot : Robot
    {
        [Parameter(DefaultValue = $strategy_name)]
        public string BotName { get; set; }
        
        // Strategy Settings
        public TradeDirection Direction; // LongOnly, ShortOnly, LongShort
        public bool ExitEncodedEntry;
        public int? ExitAfterNBars;
        public int? ExitAfterNDays;
        public bool ExitEndOfWeek;
        public bool ExitEndOfMonth;
        public double? ExitWhenPnLLessThan;
        public double? SLPips;
        public double? TPPips;
        public bool EnableTSL;
        public double TradeSize; // Could be a an Int (Units) or Double (Fraction of Liquidity) but must be converted to double volume
        
        // Private Properties
        private int _barCount; // Used for ExitAfterNBars
        private bool _startCounting; // Used for ExitAfterNBars
        
        private bool _useFixedFractional; // Fixed Fractional
        private double _tradeSize; // Fixed Fractional
        
        // DataSeries
        private DataSeries Open;
        private DataSeries High;
        private DataSeries Low;
        private DataSeries Close;
        private DataSeries Volume;
        private TimeSeries Date;
        
        protected override void OnStart()
        {   
            // Initialize DataSeries
            Open = Bars.OpenPrices;
            High = Bars.HighPrices;
            Low = Bars.LowPrices;
            Close = Bars.ClosePrices;
            Volume = Bars.TickVolumes;
            Date = Bars.OpenTimes;
            
            // To Be Generated in Evoquant. These are just some examples.
            // Instantiate Strategy Settings.
            BotName = BotName;
            Direction = $direction;
            ExitEncodedEntry = $exit_encoded_entry;
            ExitAfterNBars = $exit_after_n_bars;
            ExitAfterNDays = $exit_after_n_days;
            ExitEndOfWeek = $exit_end_of_week;
            ExitEndOfMonth = $exit_end_of_month;
            ExitWhenPnLLessThan = $exit_when_pnl_lessthan;
            //SLPips = null; //TODO: SLPips & TPPips has to be a valid double and Pip units. Percentage and Points must be converted.
            //TPPips = <>; //TODO: SLPipqs & TPPips has to be a valid double and Pip units. Percentage and Points must be converted.
            EnableTSL = $enable_tsl; //TODO: Implement the TSL in Backtesting.py
            TradeSize = $trade_size; //TODO: TradeSize need to be given as Valid Double and converted to volume/units (not lots).
            
            //Debug.Assert(ExitAfterNBars is int? && ExitAfterNBars >= 1, "ExitAfterNBars must be a positive integer");
            
            // Fix TradeSize by Volume/Units for Validity from Template
            if(TradeSize < Symbol.VolumeInUnitsMin && TradeSize >= 1) 
            {
                TradeSize = Symbol.VolumeInUnitsMin;
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down);
            }
            else
            {
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down);
            }
            
            if (TradeSize > 0 && TradeSize < 1)
            {
                _useFixedFractional = true;
            }
            else 
            {
                _useFixedFractional = false;
            }
            
            Bars.BarOpened +=OnOpen;
            Positions.Opened += OnPositionsOpened;
        }
        
        // Open Bar Event-handler
        private void OnOpen(BarOpenedEventArgs obj) 
        {   
            // Check and Update Bar Counting
            if (_startCounting) // True when there is open position. 
                _barCount++; // Update Number of Bar Count. 
            
            // Run Exits
            RunExitAfterNBars();
            RunExitAfterNDays(obj);
            RunExitWhenPnLLessThan();
            
            // Check and Update Fixed Fractional Size
            if (_useFixedFractional)
            {
                double currentSymbolPrice = obj.Bars.OpenPrices.LastValue;
                Asset SymbolQuoteCurrency = Symbol.QuoteAsset;
                _tradeSize = TradeSize * Account.Balance; // Trade Value based on current state of Balance (Account's Currency)
                _tradeSize = Account.Asset.Convert(SymbolQuoteCurrency, _tradeSize); // In Symbol's Quote Currency
                _tradeSize = Math.Floor(_tradeSize/currentSymbolPrice); // Trade Size in Volume
                _tradeSize = Symbol.NormalizeVolumeInUnits(_tradeSize, RoundingMode.Down); // Normalize to tradable amount in volume
            }
            
            // Check and Update Fixed Stop-Loss (Percentage, Pips, Points)
            SLPips = $stop_loss;
			TPPips = $take_profit;
            
            // Run Entries
            RunEntries();
        }
        
        protected override void OnTick()
        {
            // Handle price updates here
        }
        
        protected override void OnStop()
        {
            // Handle cBot stop here
        }
        //====================================Testing====================================
        
        //===============================================================================
        private void RunEntries()
        {   
            // Run Entry Code Block
            $indicator_signal_code
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            // Entry
            switch (Direction)
            {
                case TradeDirection.LongOnly:
                if (countPositions > 0) // Return if we have existing long position.
                {
                    Debug.Assert(countPositions == 1, "There is more than 1 position.");
                    // If the root signal is false, check if ExitEncodedEntry is enabled or disabled
                    if(ExitEncodedEntry & !root_signal) // ExitEncodedEntry is Enabled
                    {
                        // Close the position
                        positions[0].Close();
                        break;
                    }else
                    {
                        break;
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                if (root_signal) // True for Long Signal
                {
                    ExecuteMarketOrder(TradeType.Buy, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                break;
                //----------
                case TradeDirection.ShortOnly:
                if (countPositions > 0)
                {
                    Debug.Assert(countPositions == 1, "There is more than 1 position.");
                    // If the root signal is false, check if ExitEncodedEntry is enabled or disabled
                    if(ExitEncodedEntry & !root_signal) // ExitEncodedEntry is Enabled and root_signal is false
                    {
                        // Close the position
                        positions[0].Close();
                        break;
                    }else
                    {
                        break;
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                if (root_signal) // True for Short Signal
                {
                    ExecuteMarketOrder(TradeType.Sell, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                
                break;
                //----------
                case TradeDirection.LongShort:
                // We can only have 1 trade at a time. When one direction signal is triggered, the other has to be closed by default.
                if (countPositions > 0) // If there is an existing position (long or short)
                {
                    Debug.Assert(countPositions == 1, "There are more than 1 position");
                    // See if the open position is a long or short.
                    if(positions[0].TradeType == TradeType.Buy) // Current open position is Long
                    {
                        // Check the signals
                        if(root_signal) // True for Long
                        {
                            // Return the method because the signal is long and we have open long position.
                            break;
                        }else // False for Short
                        {
                            if(ExitEncodedEntry) // Check if ExitEncodedEntry is enabled
                            {
                                // Close the long position and place a short market order
                                positions[0].Close();
                            }else
                            {
                                break; // Break if ExitEncodedEntry not enabled
                            }
                        }
                    }else // Current open position is Short
                    {
                        // Check the signals
                        if (root_signal) // True for Long
                        {
                            if(ExitEncodedEntry) // Check if ExitEncodedEntry is enabled
                            {
                                // Close the short position and place a long market order
                                positions[0].Close();
                            }else
                            {
                                break; // Break if ExitEncodedEntry not enabled
                            }
                        }else // False for Short
                        {
                            // Return the method because the signal is short and we have open short position.
                            break;
                        }
                    }
                }
                
                Debug.Assert(countPositions == 0, "There is at least 1 position and didn't exit the method");
                // If there are no open position, see which signal is triggered. True for Long and False for Short.
                // We always check the signal.
                if (root_signal) // True for Long
                {
                    ExecuteMarketOrder(TradeType.Buy, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }else // False for Short
                {
                    ExecuteMarketOrder(TradeType.Sell, Symbol.Name, _tradeSize, BotName, SLPips, TPPips, "", EnableTSL);
                }
                break;
            }
        }
        //====================================Exit Methods====================================
        private void RunExitAfterNBars()
        {
            if(ExitAfterNBars == null) {return;} // Return the method if ExitAfterNBars is disabled
            // We Assert that ExitAfterNBars is a valid positive integer
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (countPositions == 0) {return;} // No Position. There is nothing to exit.
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            
            Position position = positions[0];
            
            if (_barCount >= ExitAfterNBars)
            {
                position.Close();
            }
        }
        
        private void RunExitAfterNDays(BarOpenedEventArgs obj)
        {
            if(ExitAfterNDays == null) {return;} // Return the method if ExitAfterNDays is disabled
            // We Assert that ExitAfterNDays is a valid positive integer
            
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (countPositions == 0) {return;} // No Position. There is nothing to exit.
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            
            Position position = positions[0];
            
            var NowDatetime = obj.Bars.OpenTimes.LastValue;
            var DateDiff = NowDatetime.Subtract(position.EntryTime).Days;
            if (DateDiff >= ExitAfterNDays)
            {
                position.Close();
            }
        }
        
        private void RunExitWhenPnLLessThan()
        {
            /*Ideally, this should be in OnTick Event-Handler*/
            var positions = Positions.FindAll(BotName, Symbol.Name); // Get all open positions
            var countPositions = Positions.Count; // Count how many open positions
            
            if (ExitWhenPnLLessThan == null) {return;} //Return method if ExitWhenPnLLessThan is disabled
            if (countPositions == 0) {return;} //Return method when there are no open positions to close
            
            Debug.Assert(countPositions == 1, "There should be only 1 position at a time");
            Position position = positions[0];
            
            // Close the position if the net profit is less than a given fixed dollar amount threshold (must be negative).
            if (position.NetProfit <= ExitWhenPnLLessThan) {position.Close();}
        }
        
        private void RunExitEndOfWeek()
        {
            // To be implemented in the future
        }
        private void RunExitEndOfMonth()
        {
            // To be implemented in the future
        }
        //====================================Other Event Handlers====================================
        void OnPositionsOpened(PositionOpenedEventArgs obj)
        {
            _startCounting = true; // Start Counting Number of bars after a position is opened.
            _barCount = 0; // Set intially to 0.
            // Remind that when a position is opened, its on the Open Price of the Bar.
        }
        
        void OnPositionsClosed(PositionClosedEventArgs obj)
        {
            // Reset Number of Bars Counting.
            _startCounting = false;
        }
        
        //====================================Signal Methods====================================
        private bool SeriesCrossAboveSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) > Series2.Last(1+0) & Series1.Last(1+1) < Series2.Last(1+1);
            return result;
        }
        private bool SeriesCrossBelowSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) < Series2.Last(1+0) & Series1.Last(1+1) > Series2.Last(1+1);
            return result;
        }
        
        private bool SeriesIsAboveSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) > Series2.Last(1+0);
            return result;
        }
        private bool SeriesIsBelowSeries(DataSeries Series1, DataSeries Series2)
        {
            bool result = Series1.Last(1+0) < Series2.Last(1+0);
            return result;
        }
        
        private bool IsIncr(DataSeries Series, int Period)
        {   
            List<Boolean> bools = new List<Boolean>();
            for (int i = 0; i < Period; i++)
            {
                bools.Add(Series.Last(1+i) > Series.Last(1+i+1));
            }
            bool AllTrue = bools.All(x => x);
            if(AllTrue) {return true;}
            else {return false;}
        }
        private bool IsDecr(DataSeries Series, int Period)
        {   
            List<Boolean> bools = new List<Boolean>();
            for (int i = 0; i < Period; i++)
            {
                bools.Add(Series.Last(1+i) < Series.Last(1+i+1));
            }
            bool AllTrue = bools.All(x => x);
            if(AllTrue) {return true;}
            else {return false;}
        }
        
        private bool IsHighest(DataSeries Series, int Period)
        {
            var CurrentValue = Series.Last(1+0); // Assume that Current is the Max
            for (int i = 1; i < Period; i++)
            {   
                var PrevVal = Series.Last(1+i);
                if(PrevVal > CurrentValue) // We found a past value higher than the current
                {
                    return false;
                }
            }
            return true;
        }
        private bool IsLowest(DataSeries Series, int Period)
        {
            var CurrentValue = Series.Last(1+0); // Assume that Current is the Min
            for (int i = 1; i < Period; i++)
            {   
                var PrevVal = Series.Last(1+i);
                if(PrevVal < CurrentValue) // We found a past value higher than the current
                {
                    return false;
                }
            }
            return true;
        }
        //====================================Utils====================================
        // Convertion Methods
    }
}
"""




__all__ = ["PTTranslator", "CTraderTranslator"]