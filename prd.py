from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *

from embedding import *
from train import *

conf = SparkConf().setAppName("testSpark").setMaster('yarn')
sc = SparkContext(conf=conf)
ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

latent = model_default.pairwise_interactions
print('The first latent row {}'.format(latent[0]))

save_data(latent, LATENT_PATH)
print('Already write latent vectors into {}.'.format(LATENT_PATH))

t0 = time.time()
new_data = latent_embedding(PRD_PATH, latent)
save_data(new_data, EMBEDDING_PATH)
t1 = time.time()
print('The first row of embeding data {}'.format(new_data[0]))
print('Successfully saved embedding data into {}. Take {} sec.'.format(EMBEDDING_PATH, t1-t0))

embedding_dim = len(new_data[0])
latent_dim = len(latent[0])
record_num = len(latent)
print('embedding data dim is %s' % embedding_dim)
print('latent vector dim is %s' % latent_dim)
print('total data length is %s' % record_num)
print('latent embedding feature dim is %s' % embedding_dim)


schema = StructType([StructField("order_id", StringType(), True)])
for i in range(table_col_num):
    schema.add("f%d" % i, DoubleType(), True)


def create_sql(table, cols, types):
    head = 'create table if not exists ' + table + '( '
    body = []
    for i, j in zip(cols, types):
        body.append('{} {}'.format(i, j))
    body = ','.join(body)
    tail = ') partitioned by (dt string)'
    return head + body + tail


embedding_table = TABLE_NAME
embedding_cols = ['order_id'] + ['f{}'.format(i) for i in range(embedding_dim)]
embedding_type = ['string'] + ['double'] * embedding_dim
embedding_sql = create_sql(embedding_table, embedding_cols, embedding_type)
ss.sql(embedding_sql)


latent_table = 'temp_jrd.pre_credit_user_fm_latent'
latent_cols = ['dim{}'.format(i+1) for i in range(latent_dim)]
latent_types = ['double'] * latent_dim
latent_sql = create_sql(latent_table, latent_cols, latent_types)
ss.sql(latent_sql)


# BATCH_SIZE = 1000
rdd = sc.parallelize(new_data)
print(rdd.first())
embedding_df = ss.createDataFrame(rdd, schema)
print(embedding_df.first())
embedding_df.registerTempTable("dataset1")
sql1 = "insert into table temp_jrd.pre_credit_user_fm_embedding partition(dt = 20170704) select * from dataset1"
ss.sql(sql1)

latent_df = ss.createDataFrame(latent)
latent_df.registerTempTable('dataset2')
sql2 = "insert into table temp_jrd.pre_credit_user_fm_latent partition(dt = 20170704) select * from dataset2"
ss.sql(sql2)
sc.stop()
# df.write.mode("append").insertInto(table_name)
