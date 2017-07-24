from __future__ import print_function
import pyFM
from prepareData.fm_index import *

label, _ = feature_target_split(os.path.join(DATA_DIR, TEST))


@clock("Take {:.4f} sec  ")
def train(method='mcmc', dim_k=8, learn_rate=0.01,
          regularization=0.01, init_std=0.1, iter_num=100, silent=True):
    fm = pyFM.FM(
        task='classification',
        learning_method=method,
        num_iter=iter_num,
        k2=dim_k,
        learn_rate=learn_rate,
        r2_regularization=regularization,
        init_stdev=init_std,
        silent=silent)
    model = fm.run(train_set=TRAIN_PATH, test_set=TEST_PATH)
    return model


def metrics(model_name):
    from sklearn.metrics import roc_auc_score, log_loss
    y_pred = model_name.predictions
    y_true = label
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print("auc:{:.6f} logloss:{:.6f}".format(auc, logloss))
    return auc


print('the default parameter model', end='')
model_default = train(silent=False)
metrics(model_default)


def grid_search(method, iter_num, dim_k, learn_rate=(0.01,), regularization=0.01, init_std=0.05):
    """
    grid search for tuning the parameter 
    :param method: 'mcmc', 'sgd' or 'als'
    :param iter_num: list object [100, 200, 300, 400]
    :param dim_k: list object
    :param learn_rate: 
    :param regularization: 
    :param init_std: 
    """
    auc_best = 0
    para_best = None
    model_best = None
    assert type(iter_num) == list
    assert type(dim_k) == list
    assert type(learn_rate) == list
    para_group = ((i, j, l) for i in dim_k for j in learn_rate for l in iter_num)  # generator
    print('\nThe training method is {}'.format(method.upper()))
    for i, j, l in para_group:
        para = 'dim k:{} learning rate:{} iter:{} '.format(i, j, l)
        print(para, end=' ')
        model = train(method=method, dim_k=i, learn_rate=j,
                      regularization=regularization, init_std=init_std, iter_num=l)
        auc = metrics(model)
        if auc > auc_best:
            auc_best = auc
            para_best = para
            model_best = model
    print('\nThe best parameter combination is:\n{} reach AUC {:.5}'.format(para_best, auc_best))
    return model_best


model_best = grid_search('mcmc', iter_num=range(100, 500, 100), dim_k=range(2, 10, 2), learn_rate=[0.01])
