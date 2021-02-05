from src.data.features import encode_by_replace, ordinal_encode_col, extract_dt
from src.data.cleaning import drop_missing_by_row
from src.rossman import drop_no_sales_days


def store_pipeline(store):
    store = store.fillna(-1)

    promo_encoding = {
        'Jan,Apr,Jul,Oct': 0,
        'Feb,May,Aug,Nov': 1,
        'Mar,Jun,Sept,Dec': 2
    }
    store = encode_by_replace(store, 'PromoInterval', promo_encoding)

    label_cols = ['StoreType', 'Assortment']
    for col in label_cols:
        store, _ = ordinal_encode_col(store, col)
    return store


def clean_pipeline(df, store):
    df = drop_no_sales_days(df)
    df = df.merge(store, on='Store')
    return drop_missing_by_row(df, verbose=False)


def feature_pipeline(tr, te, ordinal_cols=None):
    if ordinal_cols:
        for col in ordinal_cols:
            tr, enc = ordinal_encode_col(tr, col)
            te.loc[:, col] = enc.transform(te.loc[:, col].to_frame())
    tr = extract_dt(tr)
    te = extract_dt(te)
    return tr, te


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor

    from src.data.io import load_csvs
    from src.ml.split import split_features_target
    from src.ml.evaluate import evaluate_model
    from src.rossman import metric

    raw = load_csvs('raw')
    print(f'loaded data - {raw.keys()}')
    print('\nprocessing store')
    store = store_pipeline(raw['store.csv'].copy())

    #  unlike above we don't split here - we want to train on everything
    print('processing train')
    train = clean_pipeline(raw['train.csv'].copy(), store)
    print('processing holdout')
    holdout = clean_pipeline(raw['holdout.csv'].copy(), store)

    train, holdout = feature_pipeline(train, holdout, ordinal_cols=['StateHoliday'])
    final = split_features_target(train, holdout, 'Sales')

    #  check that we haven't dropped any rows in our holdout set
    assert holdout.shape[0] == drop_no_sales_days(raw['holdout.csv']).shape[0]

    print('\ntraining random forest regressor - params:')
    mdl = RandomForestRegressor(n_estimators=100, n_jobs=4, verbose=1, random_state=42)
    print(mdl.get_params())
    mdl.fit(final['x_tr'], final['y_tr'])

    print('\nevaluating model')
    evaluate_model(mdl, final, metric)
