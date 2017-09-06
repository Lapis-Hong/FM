# coding: utf-8

"""path and table config 必改参数"""
FROM_HDFS_TRAIN = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credict_user_fm/train/20170830'
FROM_HDFS_PRD = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credict_user_fm/prd/20170830'

TO_HDFS_TRAIN = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm_embedding/train/20170906'
TO_HDFS_PRD = 'hdfs://bipcluster/spark/vipshop/vipjrd/pre_credit_user_fm_embedding/prd/20170906'

# table name  must change the tablename, or delete the exist table in hive
EMBEDDING_TABLE = 'temp_jrd.pre_credit_user_fm_embedding_20170906'
LATENT_TABLE = 'temp_jrd.pre_credit_user_fm_latent_20170906'

# original libsvm train prd file
ORIGIN_TRAIN = 'fmtrain20170830'
ORIGIN_PRD = 'fmprd20170830'

# FM format total dataset
FM_TRAIN = 'train20170830'

"""path and table config 选改参数"""
# FM train test dataset
TRAIN = 'train'
TEST = 'test'

# directory
DATA_DIR = 'Data'
MODEL_DIR = 'Model'
EMBEDDING_DIR = 'Embedding'

ORIGIN_TABLE = 'temp_jrd.pre_credit_user_feature'
FROM_DT = '20170701'
TO_DT = '20170731'
DT = '20170901'

"""parameter config for FM training 选改参数"""
# support for two packages {'libfm', 'fastfm'}
PACKAGE = 'libfm'
# support for three learning method {'mcmc', 'als', 'sgd'}
METHOD = 'mcmc'
NUM_ITER = 50
DIM = 8
INIT_STDEV = 0.1
# only sgd and als need regularization params
R0_REG = 0.1
R1_REG = 0.1
R2_REG = 0.1
# only sgd need this param
LEARN_RATE = 0.1
# train set fraction
TRAIN_RATIO = 0.8


"""parameter config for FM embedding 选改参数"""
DATA_PROCESS = 'python'  # data process merhod chose from{'spark', 'shell', 'python'}
THRESHOLD = 564
# 1 means True, 0 means False  WRITE_HIVE --whether to write to hive or not, same with HDFS
ISAPPEND = 1
WRITE_HIVE = 1
WRITE_HDFS = 1
KEEP_LOCAL = 1
BATCH_SIZE = 10000

"""Spark config"""
DRIVER_MEMORY = '20g'
EXECUTOR_MEMORY = '10g'
DRIVER_CORES = '20'
CORES_MAX = '120'

