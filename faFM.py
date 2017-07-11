from fastFM import als, mcmc, sgd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from prepareData.fm_index import *
from scipy.sparse import coo_matrix

target, feature = feature_target_split(LIBFM_TRAIN)
#print(feature)

feature = coo_matrix(feature[:20])
print(target[0])
print(feature[0])

train_X, test_X, train_y, test_y = train_test_split(train=feature,
                                                    target=target,
                                                    test_size=0.2,
                                                    random_state=0)


fm = als.FMRegression(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=None)
fm.fit(train_X, train_y)
y_pred = fm.predict(test_X)


fm = sgd.FMClassification(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0, l2_reg_V=0, l2_reg=None, step_size=0.1)
fm.fit(train_X, train_y)
y_pred = fm.predict(test_X)
y_pred_proba = fm.predict_proba(test_X)

'acc:', accuracy_score(test_y, y_pred)
'auc:', roc_auc_score(test_y, y_pred_proba)


fm = mcmc.FMClassification(n_iter=100, init_stdev=0.1, rank=8, random_state=123, copy_X=True)
y_pred = fm.fit_predict(train_X, train_y, test_X)
y_pred_proba = fm.fit_predict_proba(train_X, train_y, test_X)
'acc:', accuracy_score(test_y, y_pred)
'auc:', roc_auc_score(test_y, y_pred_proba)

