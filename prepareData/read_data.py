import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from pyspark import *
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *

table = 'temp_jrd.pre_credit_user_feature_transformed_fm'

conf = SparkConf().setAppName("testSpark").setMaster('yarn').set('spark.executor.memory', '10g')
sc = SparkContext(conf=conf)
ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
spark_df = ss.sql('select * from ' + table + ' where dt=20170704')
pandas_df = spark_df.toPandas()
# data = pandas_df.to_csv('fmtrain20170704', index=False, encoding='utf-8')

# schema = StructType([StructField("order_id", StringType(), True)])
# for i in range(table_col_num):
#     schema.add("f%d" % i, DoubleType(), True)


# using rdd to remove zero values

y = pandas_df.target
X = pandas_df[np.setdiff1d(pandas_df.columns,['target'])]
# dummy = pd.get_dummies(pandas_df)
# mat = dummy.as_matrix()
dump_svmlight_file(X, y, 'fmtrain.libfm', zero_based=True, multilabel=False)

