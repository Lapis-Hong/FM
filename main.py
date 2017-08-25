try:
    import cPickle as pickle
except ImportError:
    import pickle
from pyspark import *
from conf import *
from data_process.shell import relabel_and_remove_zero
from data_process.util import get_new_index, _make_path
from train import train
from embedding import *

index_dic = get_new_index(ORIGIN_TRAIN)
pickle.dump(index_dic, open('index_dump', 'wb'))
_make_path(DATA_DIR)
_make_path(MODEL_DIR)

relabel_and_remove_zero(ORIGIN_TRAIN, FM_TRAIN)

model = train(method='mcmc',
              dim_k=8,
              iter_num=100,
              learn_rate=0.01,
              regularization=0.01,
              init_std=0.1,
              silent=False)
latent_vec = model.pairwise_interactions
pickle.dump(latent_vec, open(os.path.join(MODEL_DIR, 'latent_dump'), 'wb'))

latent, latent_dim = load_latent()


index_mapping = pickle.load(open('dump', 'rb'))
print(index_mapping)

#latent_to_hive(latent, latent_dim)
#generator_local = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
#embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=os.path.join(MODEL_DIR, EMBEDDING))

# generator_local2 = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
# embedding_to_local_generator(generator_local2, outfile=os.path.join(MODEL_DIR, EMBEDDING))

#generator_hdfs = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
#embedding_to_hdfs(generator_hdfs, hdfs_file=TO_HDFS_PATH)

generator_hive = embedding_generator(latent, infile=ORIGIN_TRAIN)
embedding_to_hive(generator_hive)

sc.stop()
ss.stop()






