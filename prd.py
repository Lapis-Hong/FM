import gc
from sklearn.datasets import dump_svmlight_file
from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *

from conf import *
from pyFM import libfm
from train import train
from embedding import *

# get latent vectors
model_default = train(silent=False)
latent = model_default.pairwise_interactions
print('The first latent row: {0}'.format(latent[0]))
latent_dim = len(latent[0])
print('latent vector dim is %s' % latent_dim)

index_mapping = pickle.load(open('dump', 'rb'))
# print(index_mapping)


def create_sql(table, cols, types):
    head = 'create table if not exists ' + table + '( '
    body = []
    for i, j in zip(cols, types):
        body.append('{} {}'.format(i, j))
    body = ','.join(body)
    tail = ') partitioned by (dt string)'
    return head + body + tail


conf = SparkConf().setAppName("testSpark").setMaster('yarn')
sc = SparkContext(conf=conf)
ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
print('Spark hive context is ready')


def latent_to_hive():
    latent_cols = ['dim{0}'.format(i+1) for i in range(latent_dim)]
    latent_types = ['string'] * latent_dim
    latent_sql = create_sql(LATENT_TABLE, latent_cols, latent_types)
    ss.sql(latent_sql)
    print('Already create {0} in hive'.format(LATENT_TABLE))

    latent_df = ss.createDataFrame(latent)
    latent_df.registerTempTable('dataset2')
    ss.sql("insert into table temp_jrd.pre_credit_user_fm_latent partition(dt = {0}) select * from dataset2".format(DT))
    print('Successfully write latent vector to hive!')


@clock('Successfully write all embedding data to hive!')
def embedding_to_hive(infile=PRD_PATH, batch_size=1000):
    new_data_generator = embedding_generator(infile, latent)
    df = []
    for lineno, elem in enumerate(new_data_generator):
        if lineno == 0:
            embedding_dim = len(elem)
            print('The first row of embedding data: {0}'.format(elem))
            print('embedding feature dim is %s' % embedding_dim)
            schema = StructType([StructField("order_id", StringType(), True)])
            for i in range(embedding_dim - 1):
                schema.add("f %d" % i, StringType(), True)
            df.append(elem)

            embedding_table = TABLE_NAME
            embedding_cols = ['order_id'] + ['f{}'.format(i) for i in range(embedding_dim - 1)]
            embedding_type = ['string'] * embedding_dim
            embedding_sql = create_sql(embedding_table, embedding_cols, embedding_type)
            ss.sql(embedding_sql)  # create embedding table in hive
            print('Successfully write latent vector to hive!')
        elif (lineno + 1) % batch_size != 0:
            df.append(elem)
        else:
            df.append(elem)
            rdd = sc.parallelize(df)
            print('the first rdd %s' % rdd.first())
            embedding_df = ss.createDataFrame(rdd, schema)
            print('the first embedding feature {0}'.format(embedding_df.first()))
            embedding_df.registerTempTable("dataset1")
            ss.sql("insert into table temp_jrd.pre_credit_user_fm_embedding partition(dt = {0}) select * from dataset1".format(DT))
            print('Already write {0} embedding data to hive'.format(lineno))
            del df
            gc.collect()
            df = []


@clock()
def embedding_to_hdfs(infile=PRD_PATH, batch_size=1000):
    new_data_generator = embedding_generator(infile, latent)
    df = []
    for lineno, elem in enumerate(new_data_generator):
        if lineno == 0:
            embedding_dim = len(elem)
            print('The first row of embedding data: {0}'.format(elem))
            print('embedding feature dim is %s' % embedding_dim)
            df.append(elem)
        elif (lineno + 1) % batch_size != 0:
            df.append(elem)
        else:
            df.append(elem)

            rdd = sc.parallelize(df)
            print('the first rdd %s' % rdd.first())
            del df
            gc.collect()
            df = []


if __name__ == '__main__':
    #latent_to_hive()
    #embedding_to_local(latent, isappend=False)
    #embedding_to_local(latent, outfile='data/libsvm',islibsvm=True)
    embedding_to_local(latent, outfile='model/temp', threshold=564)
    #embedding_to_hive()
    #embedding_to_hdfs()
sc.stop()