from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *

tbl1 = tem

conf = SparkConf().setAppName("testSpark").setMaster('yarn')
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
df = spark.sql('')


schema = StructType([StructField("order_id", StringType(), True)])
for i in range(table_col_num):
    schema.add("f%d" % i, DoubleType(), True)



