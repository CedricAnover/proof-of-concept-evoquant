import copy
import pandas as pd

from numba import njit

def extract_start_end_true(ser:pd.Series, start=True) -> pd.Series:
    assert ser.dtype == bool
    # out_s = copy.deepcopy(ser)
    s = copy.deepcopy(ser).astype(float)
    # s = s.astype(float) # Convert to dtype float
    if start:
        return ser.mask(s - s.shift(1).fillna(0.) <= 0, False)
    else:
        return ser.mask(s.shift(-1).fillna(0.) - s >= 0, False)

def end_of_wmy_dates(in_df:pd.DataFrame, mode:str) -> list:
    # out = pd.date_range(start=min(in_df.index), end=max(in_df.index), freq='D')
    years = list(set(in_df.index.year))
    week_nums = list(set(in_df.index.isocalendar().week))
    months = list(set(in_df.index.month))

    out_dates = []

    for year in years:
        if mode == 'week':
            for week_num in week_nums:
                temp_s = in_df.loc[((in_df.index.year==year)&(in_df.index.isocalendar().week==week_num)),:].index.to_series()
                if len(temp_s) < 1:
                    continue
                else:
                    out_date = max(temp_s)
                    out_dates.append(out_date.date())
        if mode == 'month':
            for m in months:
                temp_s = in_df.loc[((in_df.index.year == year) & (in_df.index.month == m)),:].index.to_series()
                if len(temp_s) < 1:
                    continue
                else:
                    out_date = max(temp_s)
                    out_dates.append(out_date.date())
        if mode == 'year':
            temp_s = in_df.loc[(in_df.index.year == year), :].index.to_series()
            if len(temp_s) < 1:
                continue
            else:
                out_date = max(temp_s)
                out_dates.append(out_date.date())
    return out_dates

# TODO: (Pending) Create a Modular, Fast, and General Solution for Cleaning bt.run()._trades
def backtesting_result_cleaner(in_df):
    df = copy.deepcopy(in_df)
    df['Direction'] = df['Size'].apply(lambda x: "Long" if x > 0 else "Short" if x < 0 else "Flat")
    return df


__all__ = ["extract_start_end_true", "end_of_wmy_dates", "backtesting_result_cleaner"]