import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def encode_by_replace(df, column, encoding):
    df.loc[:, column].replace(encoding, inplace=True)
    return df


def ordinal_encode_col(df, col):
    enc = OrdinalEncoder()
    df.loc[:, col] = enc.fit_transform(df.loc[:, col].to_frame())
    return df, enc


def extract_dt(df):
    #  conditional in case we run the cell again after dropping Date
    if 'Date' in df.columns:
        df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'])
        df.loc[:, 'year'] = df.loc[:, 'Date'].dt.year
        df.loc[:, 'month'] = df.loc[:, 'Date'].dt.month
        df.loc[:, 'day'] = df.loc[:, 'Date'].dt.day
        df.drop('Date', axis=1, inplace=True)
    return df
