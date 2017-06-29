from __future__ import print_function
import pyFM
from embedding import *
from prepareData.fm_index import *
from prepareData.embeddingIndexInit import *

LIBFM_DATA_FILE = ['fmtrain20170403.libfm'
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170403.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm'
                   ]
OUTPUT_DIR = 'output'
LATENT_FILE = 'latent'
EMBEDDING_FILE = 'embeddingData'

MODEL_PATH = 'model'  # save the FM original model result
LATENT_PATH = gen_path(OUTPUT_DIR, LATENT_FILE)
EMBEDDING_PATH = gen_path(OUTPUT_DIR, EMBEDDING_FILE)

convert_libfm(data_libsvm_file, data_temp_file, data_libfm_file)
# trainData, testData = map(split_data, ORIGINAL_DATA_PATH, TRAIN_PATH*len(), TEST_PATH)
for i in range(len(LIBFM_DATA_FILE)):
    split_data(LIBFM_DATA_FILE[i], TRAIN_FILE, TEST_FILE)
    print('%s split completed, writing train data into %s' % (LIBFM_DATA_FILE[i], TRAIN_FILE))

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
    train_set=TRAIN_FILE,
    test_set=TEST_FILE)
time_end = time.time()
print("FM training is finished. Taking %s sec." % (time_end - time_start))

# save the latent vectors
latent = model.pairwise_interactions
save_data(latent, LATENT_PATH)
print('Already write latent vectors into {}'.format(LATENT_PATH))

index_list = embedding_index_init()
new_data = latent_embedding(TRAIN_FILE, latent, index_list)
save_data(new_data, EMBEDDING_PATH)
print('Successfully saved embedding data on {}'.format(EMBEDDING_PATH))
