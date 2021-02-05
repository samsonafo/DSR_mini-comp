"""tools for looking at data quality"""
from pprint import PrettyPrinter

import pandas as pd

pp = PrettyPrinter()


def inspect_missing(df):
    """Calculates number & pct of missing values in each column in a DataFrame"""
    missing = df.isna().sum(axis=0)
    pct_missing = 100 * missing / df.shape[0]
    missing = pd.concat([missing, pct_missing], axis=1)
    missing.columns = ['num-missing', 'pct-missing']
    return missing


def inspect_uniques(df, thresh=10):
    """Looks at unique values in a column based on a threshold"""
    out = {}
    for col in df:
        uniques = set(df[col])
        if len(uniques) < 10:
            out[col] = uniques
    return out


def inspect(data, missing=False, head=0, uniques=0):
    """Interface for inspecting a DataFrame or dictionary of DataFrames"""
    if isinstance(data, dict):
        for name, df in data.items():
            print(f"{name.upper()}")
            inspect(df, missing=missing, head=head, uniques=uniques)
    else:
        inspect_single_df(data, missing=missing, head=head, uniques=uniques)


def inspect_single_df(df, missing=False, head=0, uniques=0):
    """Inspect a single DataFrame"""
    print(f"  rows: {df.shape[0]} cols: {df.shape[1]}")
    if missing:
        print('  missing values:')
        print(inspect_missing(df))
    if head:
        print(f'  head({head}):')
        print(df.head(head))

    if uniques:
        print(f"  unqiues less than {uniques}:")
        pp.pprint(inspect_uniques(df, uniques))
    print('\n')
