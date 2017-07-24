import os
import shutil

# original file path
LIBSVM_TRAIN_FILE = ['fmtrain20170403.txt']  # train data first column is target
LIBSVM_PRD_FILE = ['fm20170403.txt']  # prd data first column is order_id

# user define filename
LIBFM_TRAIN = 'fmtrain20170403.libfm'
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170403.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm'
TRAIN = 'train'  # libFM train data filename
TEST = 'test'
LIBFM_PRD = 'fm20170403.libfm'
LATENT = 'latent'  # latent vector filename
EMBEDDING = 'embeddingData'  # prd data with embedding vector filename

DATA_DIR = 'data'
OUTPUT_DIR = 'output'


def _gen_path(directory, filename):
    return os.path.join(directory, filename)


# if exists dir, remove it, os.rmdir can only remove empty dir
def _make_path(dirname):
    if os.path.exists(dirname):
        pass
        #shutil.rmtree(dirname)
    else: os.mkdir(dirname)

# MODEL_PATH = 'model'  # save the FM original model result
TRAIN_PATH = _gen_path(DATA_DIR, TRAIN)
TEST_PATH = _gen_path(DATA_DIR, TEST)
PRD_PATH = _gen_path(DATA_DIR, LIBFM_PRD)
LATENT_PATH = _gen_path(OUTPUT_DIR, LATENT)
EMBEDDING_PATH = _gen_path(OUTPUT_DIR, EMBEDDING)

_make_path(DATA_DIR)
_make_path(OUTPUT_DIR)
#_make_path(MODEL_PATH)

TABLE_NAME = 'temp_jrd.pre_credit_user_feature_fm_embedding'
