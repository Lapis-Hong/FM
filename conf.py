# coding: utf-8

"""path and table config 必改参数"""
FROM_HDFS_TRAIN = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm/train/20170830'
FROM_HDFS_PRD = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm/prd/20170830'

TO_HDFS_TRAIN = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm_embedding/train/20170830'
TO_HDFS_PRD = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm_embedding/prd/20170830'

ORIGIN_TRAIN = 'fmtrain20170830'  # original libsvm train file
ORIGIN_PRD = 'fmprd20170830'

# FM format total dataset
FM_TRAIN = 'train20170830'
# FM train dataset
TRAIN = 'train'
# FM test dataset
TEST = 'test'
# embedding data filename
EMBEDDING_TRAIN = 'embedding_train'
EMBEDDING_PRD = 'embedding_prd'

# directory
DATA_DIR = 'data'
MODEL_DIR = 'model'

# table name
EMBEDDING_TABLE = 'temp_jrd.pre_credit_user_fm_embedding'
ORIGIN_TABLE = 'temp_jrd.pre_credit_user_feature'
LATENT_TABLE = 'temp_jrd.pre_credit_user_fm_latent'

FROM_DT = '20170701'
TO_DT = '20170731'


"""parameter config for FM training 选改参数"""
# support for two packages {'libfm', 'fastfm'}
PACKAGE = 'libfm'
# support for three learning method {'mcmc', 'als', 'sgd'}
METHOD = 'mcmc'
NUM_ITER = 100
DIM = 8
INIT_STDEV = 0.1
# only sgd and als need regularization params
R0_REG = 0.1
R1_REG = 0.1
R2_REG = 0.1
# only sgd need this param
LEARN_RATE = 0.01

TRAIN_RATIO = 0.8



