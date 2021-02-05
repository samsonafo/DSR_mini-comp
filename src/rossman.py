import numpy as np


def drop_no_sales_days(df):
    no_sales_mask = df.loc[:, 'Sales'] == 0
    df = df.loc[~no_sales_mask, :]
    assert sum(df.loc[:, 'Sales'] == 0) == 0
    return df


def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
