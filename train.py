from __future__ import print_function
import os
import pyFM
from embedding import *
from prepareData.fm_index import *
from prepareData.embeddingIndexInit import *

ORIGINAL_DATA_PATH = ['/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170403.libsvm',\
                      '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm',\
                      '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm' ]
TRAIN_PATH = 'trainData' # or divide the data by dt
TEST_PATH = 'testData'


def make_path(name):
    if  os.path.exists(name):
        os.remove(name)
    os.mkdir(name)

MODEL_PATH = make_path('model')
LATENT_PATH = make_path('latent')
EMBEDDING_PATH = make_path('embeddingData')

# trainData, testData = map(split_data, ORIGINAL_DATA_PATH, TRAIN_PATH*len(), TEST_PATH)
for i in range(len(ORIGINAL_DATA_PATH)):
    split_data(ORIGINAL_DATA_PATH[i], TRAIN_PATH, TEST_PATH)
    print('%s split completed, writing train data into %s' %(ORIGINAL_DATA_PATH[i], TRAIN_PATH ))

# set the parameter of FM
fm = pyFM.FM(
    task='classification',
    num_iter=100,
    k2=8,
    learning_method='sgd',
    learn_rate=0.01,
    r2_regularization=0.01,
    temp_path=MODEL_PATH # save temp file, model path
)

# run the fm model
time_start = time.time()
model = fm.run(
    train_set = TRAIN_PATH,
    test_set = TEST_PATH)
time_end = time.time()
print("FM training is finished. Taking %s sec." %(time_end-time_start))

# save the latent vectors
latent = model.pairwise_interactions
save_data(latent, LATENT_PATH)
print('Already write latent vectors into {}'.format(LATENT_PATH))

index_list = embedding_index_init()
new_data = latent_embedding(TRAIN_PATH, latent, index_list)
save_data(new_data, EMBEDDING_PATH)
print('Successfully saved embedding data on {}'.format(EMBEDDING_PATH))





