import numpy as np
from fastFM import als, mcmc, sgd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from prepareData.fm_index import *
from scipy.sparse import coo_matrix

target, feature = feature_target_split(LIBSVM_TRAIN_FILE[0])

feature = coo_matrix(feature)
target = np.array(target)

train_X, test_X, train_y, test_y = train_test_split(feature,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0)

fm1 = als.FMClassification(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0.1)
fm1.fit(train_X, train_y)
y_pred = fm1.predict(test_X)
y_pred_proba = fm1.predict_proba(test_X)
print('acc:', accuracy_score(test_y, y_pred))
print('auc:', roc_auc_score(test_y, y_pred_proba))

fm2 = sgd.FMClassification(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0, l2_reg_V=0, l2_reg=None, step_size=0.1)
fm2.fit(train_X, train_y)
y_pred = fm2.predict(test_X)
y_pred_proba = fm2.predict_proba(test_X)

print('acc:', accuracy_score(test_y, y_pred))
print('auc:', roc_auc_score(test_y, y_pred_proba))


fm3 = mcmc.FMClassification(n_iter=100, init_stdev=0.1, rank=8, random_state=123, copy_X=True)
y_pred = fm3.fit_predict(train_X, train_y, test_X)
y_pred_proba = fm3.fit_predict_proba(train_X, train_y, test_X)
print('acc:', accuracy_score(test_y, y_pred))
print('auc:', roc_auc_score(test_y, y_pred_proba))

