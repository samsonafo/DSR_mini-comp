import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(mdl, data, metric):
    mdl.verbose = 0
    tr_pred = mdl.predict(data['x_tr'])
    te_pred = mdl.predict(data['x_te'])
    tr_score = metric(tr_pred, data['y_tr'].values)
    te_score = metric(te_pred, data['y_te'].values)
    print(f'train score {tr_score:3.2f}, test score {te_score:3.2f}')


def plot_feature_importances(rf, cols, model_dir='.'):
    importances = pd.DataFrame()
    importances.loc[:, 'importances'] = rf.feature_importances_
    importances.loc[:, 'features'] = cols
    importances.sort_values('importances', inplace=True)
    f, a = plt.subplots()
    importances.plot(ax=a, kind='bar', x='features', y='importances')
    plt.gcf().subplots_adjust(bottom=0.3)
