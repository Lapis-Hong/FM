from __future__ import print_function
import pyFM
from prepareData.fm_index import *

# run the reindex functions
main()
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
print("FM training is finished. Taking %s sec.\n" % (time_end - time_start))
# print(model.predictions)
label, _ = feature_target_split(os.path.join('data', TEST))


def metrics():
    from sklearn.metrics import roc_auc_score, log_loss
    y_pred = model.predictions
    y_true = label
    auc = roc_auc_score(y_true, y_pred)
    print('AUC is {}'.format(auc))
    logloss = log_loss(y_true, y_pred)
    print('Logloss is {}'.format(logloss))

metrics()
