from __future__ import print_function
import pyFM
from prepareData.fm_index import *

# run the reindex functions
# main()
# set the parameter of FM
fm = pyFM.FM(
    task='classification',
    num_iter=100,
    k2=8,
    learning_method='sgd',
    learn_rate=0.01,
    r2_regularization=0.01
)

# run the fm model
time_start = time.time()
model = fm.run(
    train_set=TRAIN_PATH,
    test_set=TEST_PATH)
time_end = time.time()
print("FM training is finished. Taking %.4f sec." % (time_end - time_start))
# print(model.predictions)
label, _ = feature_target_split(os.path.join(DATA_DIR, TEST))


def metrics(model_name):
    from sklearn.metrics import roc_auc_score, log_loss
    y_pred = model_name.predictions
    y_true = label
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print("auc:{:.5} logloss:{:.5}".format(auc, logloss))
    return auc

metrics(model)


def grid_search(method, iter_num, dim_k, learning_rate, regularization=0.01, init_std=0.1):
    """
    grid search for tuning the parameter 
    :param method: 'mcmc', 'sgd' or 'als'
    :param iter_num: list object [100, 200, 300, 400]
    :param dim_k: list object
    :param learning_rate: 
    :param regularization: 
    :param init_std: 
    """
    auc_best = 0
    parameter = None
    assert type(iter_num) == list
    assert type(dim_k) == list
    assert type(learning_rate) == list
    para_group = ((i, j, l) for i in dim_k for j in learning_rate for l in iter_num)  # generator
    print('The training method is {}'.format(method.upper()))
    for i, j, l in para_group:
        fm = pyFM.FM(
            task='classification',
            learning_method=method,
            num_iter=l,
            k2=i,
            learn_rate=j,
            r2_regularization=regularization,
            init_stdev=init_std,
            silent=True)
        t0 = time.time()
        model = fm.run(
            train_set=TRAIN_PATH,
            test_set=TEST_PATH)
        t1 = time.time()
        info = 'dim k:{} learning rate:{} iter:{} '.format(i, j, l)
        print(info+'  Taking %.4f sec.\n' % (t1 - t0), end='')
        auc = metrics(model)
        if auc > auc_best:
            auc_best = auc
            parameter = info
    print('The best parameter combination is:\n{} reach AUC {:.5}'.format(parameter, auc_best))


grid_search('mcmc', iter_num=range(100,500,100), dim_k=range(2,10,2), learning_rate=[0.01])
