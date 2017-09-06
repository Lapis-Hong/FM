"""Train the model and select the best model, support two packages, libfm and fastfm
Usage: python train.py -h
"""
import os
import argparse

from sklearn.metrics import roc_auc_score, log_loss

from libs import *
from conf import *
from data_process import *
from data_process.util import get_target, split_data_fastfm


@clock("FM training is finished.")
def train(package=PACKAGE, method=METHOD, dim_k=DIM, iter_num=NUM_ITER, init_stdev=INIT_STDEV,
          learn_rate=LEARN_RATE, r0_reg=R0_REG, r1_reg=R1_REG, r2_reg=R2_REG, silent=True):
    params_dict = dict(learning_method=method,
                       num_iter=iter_num,
                       k2=dim_k,
                       learn_rate=learn_rate,
                       r0_regularization=r0_reg,
                       r1_regularization=r1_reg,
                       r2_regularization=r2_reg,
                       init_stdev=init_stdev,
                       model_path=MODEL_DIR)
    assert package in ['libfm', 'fastfm'], 'package name must be one of {libfm, fastfm}'
    if package == 'libfm':
        fm_ = fm.Libfm(silent=silent, **params_dict)
        model = fm_.run(train_set=os.path.join(DATA_DIR, TRAIN), test_set=os.path.join(DATA_DIR, TEST))
    elif package == 'fastfm':
        global train_X, train_y, test_X, test_y  # if train_X... has exists, then skip the split data process
        if 'train_X' not in globals().keys():
            train_X, test_X, train_y, test_y = split_data_fastfm(ORIGIN_TRAIN, train_ratio=0.5)
        fm_ = fm.Fastfm(**params_dict)
        model = fm_.run(train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y)
    return model


def metrics(model_name):
    y_pred = model_name.predictions
    y_true = get_target(os.path.join(DATA_DIR, TEST))
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print("auc:{0:.6f} logloss:{1:.6f}\n".format(auc, logloss))
    return auc


@keyword_only
def grid_search(package=PACKAGE, method=METHOD, dim_k=DIM, iter_num=NUM_ITER, init_stdev=INIT_STDEV,
                learn_rate=LEARN_RATE, r0_reg=R0_REG, r1_reg=R1_REG, r2_reg=R2_REG, silent=True):
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
          .format(method.upper(), learn_rate, init_stdev, (r0_reg, r1_reg, r2_reg)))
    for ind, (i, j) in enumerate(para_group):
        para = '({0}) dim k:{1} iter:{2} '.format(ind+1, i, j)
        print(para)
        model = train(package=package, method=method, dim_k=i, iter_num=j,
                      init_stdev=init_stdev, learn_rate=learn_rate,
                      r0_reg=r0_reg, r1_reg=r1_reg, r2_reg=r2_reg)
        auc = metrics(model) if package == 'libfm' else model.auc
        if auc > auc_best:
            auc_best = auc
            para_best = para
            model_best = model
    print('\nThe best parameter combination is:\n{0} reach AUC {1:.5}'.format(para_best, auc_best))
    return model_best


def save_latent():
    model_default = train(silent=False)
    latent_vec = model_default.pairwise_interactions
    path = os.path.join(MODEL_DIR, 'latent_dump.libfm')
    pickle.dump(latent_vec, open(path, 'wb'))
    print('Already dump the latent vectors into {0}'.format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the FM model and tune the parameters')
    parser.add_argument("-p", "--package", default='libfm', choices=['libfm', 'fastfm'], help='fm package')
    parser.add_argument("-m", "--method", default='mcmc', choices=['sgd', 'als', 'mcmc'], help='training algorithm')
    parser.add_argument("-k", "--dim_k", type=int, default=[8], nargs='*', help='the latent dimension k')
    parser.add_argument("-n", "--iter_num", type=int, default=[100], nargs='*', help='the iteration number')
    parser.add_argument("-lr", "--learn_rate", type=float, default=0.01, help='learning rate')
    parser.add_argument("-i", "--init_stdev", type=float, default=0.1, help='inititial standard deviation of latent matrix')
    parser.add_argument("-r0", "--r0_reg", type=float, default=0.01, help='r0_regularzation')
    parser.add_argument("-r1", "--r1_reg", type=float, default=0.01, help='r1_regularzation')
    parser.add_argument("-r2", "--r2_reg", type=float, default=0.01, help='r2_regularzation')
    parser.add_argument("-v", "--silent", default=False, help='verbose mode')
    arg = parser.parse_args()
    param_dict = vars(arg)

    if len(arg.iter_num) > 1 or len(arg.dim_k) > 1:
        if arg.package == 'fastfm':
            train_X, test_X, train_y, test_y = split_data_fastfm(ORIGIN_TRAIN)
        model = grid_search(**param_dict)
    else:
        param_dict['dim_k'] = param_dict['dim_k'][0]  # turn 'dim_k':[8] to 8
        param_dict['iter_num'] = param_dict['iter_num'][0]
        model = train(**param_dict)
        if arg.package == 'libfm':
            metrics(model)
        print('The model parameters: {0}'.format(param_dict))

    latent = model.pairwise_interactions
    pickle.dump(latent, open(os.path.join(MODEL_DIR, 'latent_dump.{0}'.format(arg.package)), 'wb'))  # serialize the latent vec
