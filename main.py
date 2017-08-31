try:
    import cPickle as pickle
except ImportError:
    import pickle
from pyspark import *
from conf import *
from data_process.shell import relabel_and_remove_zero
from data_process.util import *
from train import train
from embedding import *

make_path(DATA_DIR)
make_path(MODEL_DIR)
index_dic = get_new_index(ORIGIN_TRAIN)
pickle.dump(index_dic, open('index_dump', 'wb'))

if not os.path.exists(FM_TRAIN):
    relabel_and_remove_zero(ORIGIN_TRAIN, FM_TRAIN)

model = train(silent=False)
latent_vec = model.pairwise_interactions
pickle.dump(latent_vec, open(os.path.join(MODEL_DIR, 'latent_dump.libfm'), 'wb'))

latent, latent_dim = load_latent()


index_mapping = pickle.load(open('index_dump', 'rb'))
print('index mapping: {0}'.format(index_mapping))


sc, ss = start_spark()
latent_to_hive(latent, latent_dim)

# embedding to local
embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
embedding_to_local(latent, infile=ORIGIN_PRD, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))

# train_generator_local = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
# prd_generator_local = embedding_generator(latent, infile=ORIGIN_PRD, islibsvm=True)
# embedding_to_local_generator(train_generator_local, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
# embedding_to_local_generator(prd_generator_local, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))

# embedding to hdfs
train_generator_hdfs = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
prd_generator_hdfs = embedding_generator(latent, infile=ORIGIN_PRD, islibsvm=True)
embedding_to_hdfs(train_generator_hdfs, hdfs_file=TO_HDFS_TRAIN)
embedding_to_hdfs(prd_generator_hdfs, hdfs_file=TO_HDFS_PRD)

# embedding to hive
generator_hive = embedding_generator(latent, infile=ORIGIN_PRD, hastarget=True)
embedding_to_hive(generator_hive)
sc.stop()
ss.stop()







