"""Train the model and select the best model
Usage: python train.py -h
"""
from __future__ import print_function
import os
import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pyFM import libfm
from conf import *
from data_process.util import get_target
from data_process import clock, keyword_only


@clock("FM training is finished.")
def train(method='mcmc', dim_k=8, learn_rate=0.01,
          regularization=0.01, init_std=0.1, iter_num=100, silent=True):
    fm = libfm.FM(
        task='classification',
        learning_method=method,
        num_iter=iter_num,
        k2=dim_k,
        learn_rate=learn_rate,
        r2_regularization=regularization,
        init_stdev=init_std,
        silent=silent,
        model_path=MODEL_DIR)
    model = fm.run(train_set=os.path.join(DATA_DIR, TRAIN), test_set=os.path.join(DATA_DIR, TEST))
    return model


def metrics(model_name):
    from sklearn.metrics import roc_auc_score, log_loss
    y_pred = model_name.predictions
    y_true = get_target(os.path.join(DATA_DIR, TEST))
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print("auc:{0:.6f} logloss:{1:.6f}".format(auc, logloss))
    return auc


@keyword_only
def grid_search(method, iter_num, dim_k, learn_rate=0.01, init_std=0.05, regularization=0.01):
    """
    grid search for tuning the parameter 
    :param method: 'mcmc', 'sgd' or 'als'
    :param iter_num: list type [100, 200, 300, 400]
    :param dim_k: list type
    """
    auc_best = 0
    para_best = None
    model_best = None
    if isinstance(iter_num, int):
        iter_num = [iter_num]
    if isinstance(dim_k, int):
        dim_k = [dim_k]
    para_group = ((i, j) for i in dim_k for j in iter_num)  # generator
    print('\nThe training method is {0}, learning rate: {1} initstd: {2} regularization:{3}'
          .format(method.upper(), learn_rate, init_std, regularization))
    for i, j in para_group:
        para = 'dim k:{0} iter:{1} '.format(i, j)
        print(para, end=' ')
        model = train(method=method, dim_k=i, learn_rate=learn_rate,
                      regularization=regularization, init_std=init_std, iter_num=j)
        auc = metrics(model)
        if auc > auc_best:
            auc_best = auc
            para_best = para
            model_best = model
    print('\nThe best parameter combination is:\n{0} reach AUC {1:.5}'.format(para_best, auc_best))
    return model_best

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the FM model and tune the parameters')
    parser.add_argument("-m", "--method", default='mcmc',
                        choices=['sgd', 'als', 'mcmc', 'sgda'], help='the training algorithm')
    parser.add_argument("-dim", type=int, default=8, nargs='*', help='the latent dimension k')
    parser.add_argument("-iter", type=int, default=100, nargs='*', help='the iteration number')
    parser.add_argument("-lr", "--learn_rate", type=int, default=0.01, help='learning rate')
    parser.add_argument("-i", "--init_stdev", type=int, default=0.1,
                        help='the inititial standard deviation of the latent matrix')
    parser.add_argument("-r", type=int, default=0.01, help='regularzation')
    parser.add_argument("-v", "--verbose", default=False, help='verbose mode')
    arg = parser.parse_args()
    if isinstance(arg.iter, list) or isinstance(arg.dim, list):  # arg.iter = [100 ,200]
        model = grid_search(arg.method, iter_num=arg.iter,
                            dim_k=arg.dim, learn_rate=arg.learn_rate,
                            regularization=arg.r, init_std=arg.init_stdev)
    else:
        model = train(method=arg.method, dim_k=arg.dim, learn_rate=arg.learn_rate,
                      regularization=arg.r, init_std=arg.init_stdev,
                      iter_num=arg.iter, silent=arg.verbose)
        metrics(model)
        print('The model parameters: {}'.format(arg))

    latent = model.pairwise_interactions
    pickle.dump(latent, open(os.path.join(MODEL_DIR, 'latent_dump'), 'wb'))  # serialize the latent vec
