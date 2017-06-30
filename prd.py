# from embedding import *
from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from prepareData.embeddingIndexInit import *
from embedding import *
from train import *

# libfm_prd_file = '/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fm20170405.libsvm'
index_list = embedding_index_init()
# save the latent vectors
latent = model.pairwise_interactions
save_data(latent, LATENT_PATH)
print('Already write latent vectors into {}'.format(LATENT_PATH))

new_data = latent_embedding(LIBFM_PRD_FILE, latent, index_list)
save_data(new_data, EMBEDDING_PATH)
print('Successfully saved embedding data into {}'.format(EMBEDDING_PATH))

latent_dim = len(latent[0])
record_num = len(latent)
print ('latent vector dim is %s' % latent_dim)
print ('total record number is %s' % record_num)

table_col_num = len(index_list) * latent_dim
print ('latent embedding feature dim is %s' % table_col_num)


conf = SparkConf().setAppName("testSpark")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

schema = StructType([StructField("order_id", StringType(), True)])
for i in range(table_col_num):
    schema.add("f%d" % i, DoubleType(), True)


def create_sql(table_name, table_col_num):
    sql_list = ['create table if not exists ']
    sql_list.append(table_name)
    sql_list.append(' (order_id string')
    for i in range(table_col_num):
        sql_list.append(',f%d double' % i)
    sql_list.append(') partitioned by (dt string)')
    table_sql = ''.join(sql_list)
    return table_sql
table_sql = create_sql(TABLE_NAME, table_col_num)
spark.sql(table_sql)
# create the table in hive


# BATCH_SIZE = 1000
rdd = sc.parallelize(new_data)
df = spark.createDataFrame(rdd, schema)
# df.collect()
df.registerTempTable("dataSet")
sql = "insert into table temp_jrd.wphorder_used_fm_embedding partition(dt = 20170403) select * from dataSet"
spark.sql(sql)
sc.stop()


