
FROM_HDFS_PATH = 'hdfs://temp'
TO_HDFS_PATH = 'hdfs://temp'
# original file
ORIGIN_TRAIN = 'fmtrain20170403.txt'  # train data first column is target


# user define filename
FM_TRAIN = 'train20170403'
SPARK_FILE = 'train20170403_spark'
TRAIN = 'train'
TEST = 'test'
EMBEDDING = 'embedding_data'  # data with embedding vector filename

# directory
DATA_DIR = 'data'
MODEL_DIR = 'model'

# table name
EMBEDDING_TABLE = 'temp_jrd.pre_credit_user_fm_embedding'
ORIGIN_TABLE = 'temp_jrd.pre_credit_user_feature_transformed_lr'
LATENT_TABLE = 'temp_jrd.pre_credit_user_fm_latent'
DT = '20170801'


