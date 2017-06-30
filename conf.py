import os
import shutil


LIBSVM_TRAIN_FILE = 'fmtrain20170403.txt'  # train data first column is target
LIBSVM_PRD_FILE = 'fm20170403.txt'  # prd data first column is order_id
LIBFM_TRAIN_FILE = 'fmtrain20170403.libfm'
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170403.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm',
                   # '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm'
LIBFM_PRD_FILE = 'fm20170403.libfm'

TRAIN_FILE = 'train'  # libFM train data filename
TEST_FILE = 'test'
LATENT_FILE = 'latent'  # latent vector filename
EMBEDDING_FILE = 'embeddingData'  # prd data with embedding vector filename

DATA_DIR = 'data'
OUTPUT_DIR = 'output'


def gen_path(directory, filename):
    return os.path.join(directory, filename)


# if exists dir, remove it, os.rmdir can only remove empty dir
def make_path(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

MODEL_PATH = 'model'  # save the FM original model result
TRAIN_PATH = gen_path(DATA_DIR, TRAIN_FILE)
TEST_PATH = gen_path(DATA_DIR, TEST_FILE)
LATENT_PATH = gen_path(OUTPUT_DIR, LATENT_FILE)
EMBEDDING_PATH = gen_path(OUTPUT_DIR, EMBEDDING_FILE)

make_path(DATA_DIR)
make_path(OUTPUT_DIR)
make_path(MODEL_PATH)

TABLE_NAME = 'temp_jrd.wphorder_used_fm_embedding'
