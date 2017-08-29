"""Using Spark to do the data preprocess
@ cluter mode: {'yarn'} infile default read from hdfs, 
               could be local, need to specify file://absolute path, must available on all workers.
but when hdfs, got error "native snappy library not available", don't know how to fix.
@ local mode: {'local', 'local[k]', 'local[*]'} infile local
local use single thread, local[k] use k threads, much faster, local[*] auto use the cores num threads.
so here use the local[*] mode.
Performance: 8G data size take 130s to do the preprocess,
             compare with awk 700s and python 780s.
"""
import os
import subprocess

import numpy as np
from sklearn.datasets import dump_svmlight_file
from pyspark import *
from pyspark.sql import SparkSession, Row

from conf import *
from data_process import *
from data_process.python import reformat
from data_process.shell import load_data_from_hdfs
from data_process.util import *


@clock('Successfully convert to the libfm format!')
def convert_from_local_byspark(infile, outfile, isprd=False, reindex=False, keep_zero=False):
    """use spark to convert the local libsvm data to libfm"""
    index_mapping = pickle.load(open('dump', 'rb')) if reindex else None
    print('index mapping has loaded')
    conf = SparkConf().setMaster('local[*]').set('spark.driver.memory', '10g').set('spark.executor.memory', '10g')
    sc = SparkContext(conf=conf)
    inpath = os.path.abspath(infile)
    # hadoop output dir can not be exists, must be removed
    subprocess.call('hadoop fs -rm -r {}'.format(outfile), shell=True)
    rdd = sc.textFile('file://{0}'.format(inpath))
    data = rdd.map(reformat(isprd, reindex, keep_zero, index_mapping))  # passing params func to map()
    # path = os.path.join(DATA_DIR, outfile)
    # if os.path.exists(path):
    #     shutil.rmtree(path)  # if exists dir, remove it, os.rmdir can only remove empty dir
    # a lot of files in hdfs
    data.saveAsTextFile(outfile, 'org.apache.hadoop.io.compress.GzipCodec')  # can not use snappy
    # data.repartition(1).saveAsTextFile(outfile)  # can be merged into 1 file, but for big data, it would be slow
    sc.stop()


"""TODO: unfinished, need to debug"""
@clock('Successfully convert to the libfm format!')
def convert_from_hive_byspark(table, dt,  outfile, save_origin=False, origin_file=None):
    conf = SparkConf().setAppName("fm_data_process").setMaster('yarn').set('spark.executor.memory', '10g')
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
    ss.stop()


if __name__ == '__main__':
    make_path(DATA_DIR, MODEL_DIR)

    load_data_from_hdfs(FROM_HDFS_TRAIN, ORIGIN_TRAIN)
    load_data_from_hdfs(FROM_HDFS_PRD, ORIGIN_PRD)

    index_dic = get_new_index(ORIGIN_TRAIN)
    pickle.dump(index_dic, open(os.path.join(MODEL_DIR, 'index_dump'), 'wb'))

    temp_path = 'hdfs://bipcluster/user/u_jrd_lv1/fm_temp'
    convert_from_local_byspark(ORIGIN_TRAIN, temp_path)
    load_data_from_hdfs(temp_path, os.path.join(DATA_DIR, FM_TRAIN))
    split_data(FM_TRAIN, TRAIN, TEST, mode='overwrite')

