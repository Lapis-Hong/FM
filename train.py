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
    r2_regularization=0.01,
    temp_path=MODEL_PATH  # save temp file, model path
)

# run the fm model
time_start = time.time()
model = fm.run(
    train_set=TRAIN_PATH,
    test_set=TEST_PATH)
time_end = time.time()
print("FM training is finished. Taking %s sec." % (time_end - time_start))
