"""Using Spark to do the data preprocess
TODO: need to test the performance and debug"""
import shutil
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
from pyspark import *
from pyspark.sql import SparkSession, Row

from conf import *
from data_process import clock
from data_process.python import reformat


@clock('Successfully convert to the libfm format!')
def convert_from_local_byspark(infile, outfile, isprd=False, reindex=False, keep_zero=False):
    """use spark to convert the local libsvm data to libfm"""
    index_mapping = pickle.load(open('dump', 'rb')) if reindex else None
    print('index mapping has loaded')
    conf = SparkConf()#.setMaster('yarn')
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(infile)
    data = rdd.map(reformat(isprd, reindex, keep_zero, index_mapping))  # passing params func to map()
    if os.path.exists(os.path.join(DATA_DIR, SPARK_FILE)):
        shutil.rmtree(os.path.join(DATA_DIR, SPARK_FILE))
    data.repartition(1).saveAsTextFile(outfile)
    sc.stop()


"""TODO: unfinished, need to debug"""
@clock('Successfully convert to the libfm format!')
def convert_from_hive_byspark(table, dt,  outfile, save_origin=False, origin_file=None):
    conf = SparkConf().setAppName("testSpark").setMaster('yarn').set('spark.executor.memory', '10g')
    #sc = SparkContext(conf=conf)
    ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    spark_df = ss.sql('select * from ' + table + ' where dt={0}'.format(dt))
    #sc.stop()
    pandas_df = spark_df.toPandas()
    if save_origin:
        pandas_df.to_csv(origin_file, index=False, encoding='utf-8')
    y = pandas_df.target
    X = pandas_df[np.setdiff1d(pandas_df.columns, ['target'])]
    # dummy = pd.get_dummies(pandas_df)
    # mat = dummy.as_matrix()
    dump_svmlight_file(X, y, outfile, zero_based=True, multilabel=False)


if __name__ == '__main__':
    convert_from_local_byspark(ORIGIN_TRAIN, SPARK_FILE)
    #convert_from_hive_bySpark(ORIGIN_TABLE, 20170720, SPARK_FILE2)
