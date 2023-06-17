from evoquant import base, signals, indicators, evo_gp

from string import Template
import textwrap

import inspect

base_module_str = rf"{inspect.getsource(base)}"
signals_module_str = rf"{inspect.getsource(signals)}"
indicators_module_str = rf"{inspect.getsource(indicators)}"
utils_module_str = rf"{inspect.getsource(evo_gp)}"
exec(base_module_str)
exec(signals_module_str)
exec(indicators_module_str)
exec(utils_module_str)

# module_content_str = inspect.getsource(base)
# print(module_content_str)
# # exec(module_content_str)

test_str1 = \
r"""
# This should be a valid python code to be executed.

max_lag = {param_max_lag}
print("Max Lag is", max_lag)
print("Max Lag will become a global within this module.")

{some_code}
"""

test_str2 = \
r"""
print("\n")
# This is some code to be appended to test_str1
def some_func1():
    print("This is a function inside test_str2.")
    print("We are printing string {param_word}")
    print("We will use what we learned here to generate valid trading codes in cTrader, MT4/5, EasyLanguage, and other Trading Programming Languages.")

def some_func2({param_non_str_var}):
    print("Type: ", type({param_non_str_var}))
    print(str({param_non_str_var}), " becomes a variable in test_str2.")
    print(str({param_non_str_var}))

some_func1()
some_func2({param_non_str_var}=(1+1.))
"""

# Note: If you want to add a string code in another string code you have to put it in as an executed string code parameter.
# test_str1 = test_str1.format(param_max_lag=5,
#                              some_code=exec(test_str2.format(param_word='WORD', param_non_str_var='x')
#                                             )
#                              )

# This will be useful for creating several lines of Code in a target language. [Value|Expression|Variable|Function|Class|Code]
test_str1 = test_str1.format(param_max_lag=5,
                             some_code=test_str2.format(param_word='WORD', param_non_str_var='x')
                             )
exec(test_str1)
print(test_str1)

