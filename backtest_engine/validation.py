import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

import datetime
from typing import Tuple, Union

def linear_is_oos(in_df:Union[pd.Series, pd.DataFrame], test_size:float=0.4, date_only:bool=True) -> Tuple[Tuple[datetime.date, datetime.date], Tuple[datetime.date, datetime.date]]:
    # Assume: in_df is clean and has date/datetime index

    # Convert the date index to a NumPy array
    date_index = in_df.index.to_numpy().reshape(-1, 1)

    # Perform train-test split
    is_dates, oos_dates = train_test_split(date_index, test_size=test_size, shuffle=False)

    # Retrieve train and test start/end dates
    is_start_date = pd.to_datetime(is_dates[0][0])
    is_end_date = pd.to_datetime(is_dates[-1][0])
    oos_start_date = pd.to_datetime(oos_dates[0][0])
    oos_end_date = pd.to_datetime(oos_dates[-1][0])

    if date_only:
        # IS(start, end) & OOS(start, end)
        return (is_start_date.date(), is_end_date.date()), (oos_start_date.date(), oos_end_date.date())
    else:
        # IS(start, end) & OOS(start, end)
        return (is_start_date, is_end_date), (oos_start_date, oos_end_date)

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits, train_ratio, margin=0):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.margin = margin
    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_ratio * (stop - start)) + start
            yield indices[start: mid], indices[mid + self.margin: stop]

def multi_linear_is_oos(in_df:Union[pd.Series,pd.DataFrame], n_splits=2, train_ratio=.6, margin=0, date_only:bool=True):
    if (not isinstance(n_splits, int)) or (n_splits < 2): raise ValueError("n_splits must be an integer and at least 2.")

    bts = BlockingTimeSeriesSplit(n_splits, train_ratio, margin=margin)
    out_dict = dict()
    i = 0
    for train_index, test_index in bts.split(in_df.index):
        train = in_df.iloc[train_index]
        test = in_df.iloc[test_index]
        out_dict[i] = {"IS":(train.index.to_series().min().date(), train.index.to_series().max().date()),
                       "OOS":(test.index.to_series().min().date(), test.index.to_series().max().date())
                       }
        i += 1
    return out_dict # out_dict[0]["IS"][0], out_dict[0]["IS"][1], out_dict[0]["OOS"][0], out_dict[0]["OOS"][1]

# TODO: For 3rd Filter Layer, plan Packages/Modules/Classes/Functions/Variables for Monte-Carlo Methods and Walk-Forward Testing.
    # Note: WF Testing Must be Compatible with Backtesting.py


__all__ = ["linear_is_oos", "multi_linear_is_oos"]