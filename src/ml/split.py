from sklearn.model_selection import train_test_split


def simple_time_series_split(tr_te, test_size=0.1):
    tr, te = train_test_split(tr_te, test_size=test_size, shuffle=False)
    assert tr.shape[0] > te.shape[0]
    assert tr.shape[0] + te.shape[0] == tr_te.shape[0]
    assert tr.shape[1] == te.shape[1]
    assert max(tr.loc[:, 'Date']) < max(te.loc[:, 'Date'])
    return tr, te


def split_features_target(tr, te, target):
    final = {}
    final['y_tr'] = tr.loc[:, target]
    final['x_tr'] = tr.drop(target, axis=1)

    final['y_te'] = te.loc[:, target]
    final['x_te'] = te.drop(target, axis=1)

    assert final['y_tr'].shape[0] == final['x_tr'].shape[0]
    assert final['y_te'].shape[0] == final['x_te'].shape[0]
    assert final['x_tr'].shape[1] == final['x_te'].shape[1]
    return final
