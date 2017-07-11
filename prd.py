from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from embedding import *
from train import *


latent = model.pairwise_interactions
print(latent[0])

t0 = time.time()
save_data(latent, LATENT_PATH)
t1 = time.time()
print('Already write latent vectors into {}. Take {} sec.'.format(LATENT_PATH, t1-t0))

t0 = time.time()
new_data = latent_embedding(PRD_PATH, latent)
save_data(new_data, EMBEDDING_PATH)
t1 = time.time()
print(new_data[0])
print('Successfully saved embedding data into {}. Take {} sec.'.format(EMBEDDING_PATH, t1-t0))


embedding_dim = len(new_data[0])
latent_dim = len(latent[0])
record_num = len(latent)
print('embedding_dim is %s' % embedding_dim)
print('latent vector dim is %s' % latent_dim)
print('total record number is %s' % record_num)

table_col_num = embedding_dim * latent_dim
print('latent embedding feature dim is %s' % table_col_num)


conf = SparkConf().setAppName("testSpark").setMaster('yarn')
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
print(table_sql)
spark.sql(table_sql)
# create the table in hive


# BATCH_SIZE = 1000
rdd = sc.parallelize(new_data)
print(rdd.top(10))
df = spark.createDataFrame(rdd, schema)
print(df.first())

df.registerTempTable("dataSet")
sql = "insert into table temp_jrd.wphorder_used_fm_embedding partition(dt = 20170403) select * from dataSet"
spark.sql(sql)
sc.stop()


