import os
try:
    import cPickle as pickle
except:
    import pickle

# original file path
HDFS_PATH = ''
ORIGIN_TRAIN = 'fmtrain20170403.libsvm'  # train data first column is target
ORIGIN_PRD = 'fm20170403.libsvm'  # prd data first column is order_id
SPARK_FILE = 'fmtrain_spark'
SPARK_FILE2 = 'fromhive'

# user define filename
FM_TRAIN = 'fmtrain20170403.libfm'
TRAIN = 'train'
TEST = 'test'
FM_PRD = 'fm20170403.libfm'
EMBEDDING = 'embedding_data'  # prd data with embedding vector filename

DATA_DIR = 'data'
MODEL_DIR = 'model'

TABLE_NAME = 'temp_jrd.pre_credit_user_fm_embedding'
ORIGIN_TABLE = 'temp_jrd.pre_credit_user_feature_transformed_lr'
LATENT_TABLE = 'temp_jrd.pre_credit_user_fm_latent'
DT = '20170801'


def gen_path(directory, filename):
    return os.path.join(directory, filename)


# if exists dir, remove it, os.rmdir can only remove empty dir
def _make_path(dirname):
    if os.path.exists(dirname):
        pass
        #shutil.rmtree(dirname)
    else:
        os.mkdir(dirname)


TRAIN_PATH = gen_path(DATA_DIR, TRAIN)
TEST_PATH = gen_path(DATA_DIR, TEST)
PRD_PATH = gen_path(DATA_DIR, FM_PRD)
EMBEDDING_PATH = gen_path(MODEL_DIR, EMBEDDING)

if __name__ == '__main__':
    _make_path(DATA_DIR)
    _make_path(MODEL_DIR)

