try:
    import cPickle as pickle
except ImportError:
    import pickle

from pyspark import *

from conf import *
from data_process.shell import relabel_and_remove_zero
from data_process.python import convert_from_local
from data_process.spark import convert_from_local_byspark
from data_process.util import *
from train import train, save_latent
from embedding import *

# data process
make_path(DATA_DIR, MODEL_DIR)
load_data_from_hdfs(FROM_HDFS_TRAIN, ORIGIN_TRAIN)
load_data_from_hdfs(FROM_HDFS_PRD, ORIGIN_PRD)

index_dic = get_new_index(ORIGIN_TRAIN)
pickle.dump(index_dic, open(os.path.join(MODEL_DIR, 'index_dump'), 'wb'))
print('index mapping: {0}'.format(index_dic))

if DATA_PROCESS == 'shell':
    relabel_and_remove_zero(ORIGIN_TRAIN, os.path.join(DATA_DIR, FM_TRAIN))
elif DATA_PROCESS == 'spark':
    temp_path = 'hdfs://bipcluster/user/u_jrd_lv1/fm_temp'
    convert_from_local_byspark(ORIGIN_TRAIN, temp_path)
    load_data_from_hdfs(temp_path, os.path.join(DATA_DIR, FM_TRAIN))
else:
    convert_from_local(ORIGIN_TRAIN, FM_TRAIN)

split_data(FM_TRAIN, TRAIN, TEST)

# train model
save_latent()
latent, latent_dim = load_latent()

# sc, ss = start_spark()
latent_to_hive(latent, latent_dim)

# embedding to local
embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
embedding_to_local(latent, infile=ORIGIN_PRD, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))

# train_generator = embedding_generator(latent, infile=ORIGIN_TRAIN)
# prd_generator = embedding_generator(latent, infile=ORIGIN_PRD)
# embedding_to_local_generator(train_generator, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
# embedding_to_local_generator(prd_generator, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))

# embedding to hdfs
train_generator = embedding_generator(latent, infile=ORIGIN_TRAIN)
prd_generator = embedding_generator(latent, infile=ORIGIN_PRD)
embedding_to_hdfs(train_generator, hdfs_file=TO_HDFS_TRAIN)
embedding_to_hdfs(prd_generator, hdfs_file=TO_HDFS_PRD)

# embedding to hive
generator_hive = embedding_generator(latent, infile=ORIGIN_PRD, hastarget=True)
embedding_to_hive(generator_hive)
sc.stop()
ss.stop()







