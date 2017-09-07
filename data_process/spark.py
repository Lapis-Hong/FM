"""Using Spark to do the data preprocess
@ cluter mode: {'yarn'} infile default read from hdfs, 
               could be local, need to specify file://absolute path, must available on all workers.
but when hdfs, got error "native snappy library not available", don't know how to fix.
@ local mode: {'local', 'local[k]', 'local[*]'} infile local
local use single thread, local[k] use k threads, much faster, local[*] auto use the cores num threads.
so here use the local[*] mode. and for I/O dense application it maybe better.
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
from data_process.util import *


@clock('Successfully convert to the libfm format!')
def convert_from_local_byspark(infile, outfile, isprd=False, reindex=False, keep_zero=False):
    """use spark to convert the local libsvm data to libfm"""
    temp_path = 'hdfs://bipcluster/user/u_jrd_lv1/fm_temp'
    if reindex:
        index_mapping = pickle.load(open(os.path.join(MODEL_DIR, 'index_dump'), 'rb'))
        print('index mapping has loaded')
    else:
        index_mapping = None
    sc, _ = start_spark(yarn_master=False)
    inpath = os.path.abspath(infile)
    # hadoop output dir can not be exists, must be removed
    subprocess.call('hadoop fs -rm -r {0}'.format(temp_path), shell=True)
    rdd = sc.textFile('file://{0}'.format(inpath))
    data = rdd.map(reformat(isprd, reindex, keep_zero, index_mapping))  # passing params func to map()
    # path = os.path.join(DATA_DIR, outfile)
    # if os.path.exists(path):
    #     shutil.rmtree(path)  # if exists dir, remove it, os.rmdir can only remove empty dir
    # a lot of files in hdfs
    data.saveAsTextFile(temp_path, 'org.apache.hadoop.io.compress.GzipCodec')  # can not use snappy
    # data.repartition(1).saveAsTextFile(outfile)  # can be merged into 1 file, but for big data, it would be slow
    hdfs_to_local(temp_path, os.path.join(DATA_DIR, outfile))
    sc.stop()


@clock('Successfully convert to the libfm format!')
def convert_from_hive_byspark(table, dt,  outfile, save_origin=False, origin_file=None):
    """TODO: unfinished, need to debug"""
    sc, ss = start_spark()
    spark_df = ss.sql('select * from ' + table + ' where dt={0}'.format(dt))
    pandas_df = spark_df.toPandas()
    if save_origin:
        pandas_df.to_csv(origin_file, index=False, encoding='utf-8')
    y = pandas_df.target
    X = pandas_df[np.setdiff1d(pandas_df.columns, ['target'])]
    # dummy = pd.get_dummies(pandas_df)
    # mat = dummy.as_matrix()
    dump_svmlight_file(X, y, outfile, zero_based=True, multilabel=False)
    sc.stop()


def main():
    load_data()
    convert_from_local_byspark(ORIGIN_TRAIN, FM_TRAIN)
    split_data(FM_TRAIN, TRAIN, TEST)

if __name__ == '__main__':
    main()


