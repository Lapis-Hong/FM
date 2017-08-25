import sys
import os
import shutil
try:
    import cPickle as pickle
except ImportError:
    import pickle

import pandas as pd
from pyspark import *
from pyspark.sql import SparkSession

from conf import *
from data_process import *


def _make_path(dirname, overwrite=False):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    elif overwrite:
        shutil.rmtree(dirname)  # if exists dir, remove it, os.rmdir can only remove empty dir
        os.mkdir(dirname)


@clock('Index mapping is finished!')
def get_new_index(infile, eachline=False):
    """
    if the original libsvm data index is not from zero, use this function to do index mapping 
    foreachLine: if the original libsvm index is different for each line, then set True
    return: {66:0, 67:1, 75:2...}
    """
    with open(infile) as f:
        index_set = set()  # save all the original index
        for line in f:
            line = line.strip('\n').split(' ')
            line.pop(0)
            for w in line:
                index = int(w.split(':')[0])
                if index not in index_set:
                    index_set.add(index)
            if not eachline:
                break
        index_list = sorted(list(index_set))
        if sys.version < '2.7':
            dic = dict([(k, v) for v, k in enumerate(index_list)])
        else:
            dic = {k: v for v, k in enumerate(index_list)}  # mapping the index to new index
        return dic


def get_target(infile):
    with open(infile) as f:
        return [int(float(line.split(' ').pop(0))) for line in f]


@clock('Train test split has completed!')
def split_data(infile, train_file, test_file, train_ratio=0.8, chunksize=200000, mode='error'):
    """
    make train and test file for libFM train, shuffle the data
    :param infile: FM TRAIN FILE
    :param train_file: libFM train filename
    :param test_file: libFM test filename
    :param train_ratio:
    :param mode: if already exists train file, provide 3 mode to deal with it {"error", "overwrite", "append"}
    :return: the libFM train and test data path
    """
    train_path = os.path.join(DATA_DIR, train_file)
    test_path = os.path.join(DATA_DIR, test_file)
    if os.path.exists(train_path):
        if mode == 'error':
            raise IOError('train file:{} already exsits. Please change it in conf.py'.format(train_file))
        elif mode == 'overwrite':
            os.remove(train_path)
            os.remove(test_path)
        elif mode == 'append':
            pass
        else:
            raise TypeError('No such mode')
    train_length, test_length = 0, 0
    for df in pd.read_csv(os.path.join(DATA_DIR, infile), header=None, chunksize=chunksize):
        train = df.sample(frac=train_ratio)
        train_length += train.shape[0]
        train.to_csv(train_path, mode='a', index=False, header=None)
        test_index = set(df.index) - set(train.index)
        test = df.iloc[list(test_index), :]
        test_length += test.shape[0]
        test.to_csv(test_path, mode='a', index=False, header=None)
    print('Train data length: {0}'.format(train_length))
    print('Test data length: {0}'.format(test_length))
    return os.path.abspath(train_file), os.path.abspath(test_file)


def create_table(table, cols, types):
    """create table sql for hive"""
    head = 'create table if not exists ' + table + '( '
    body = []
    for i, j in zip(cols, types):
        body.append('{0} {1}'.format(i, j))
    body = ','.join(body)
    tail = ') partitioned by (dt string)'
    return head + body + tail


def start_spark(master='yarn'):
    if master == 'yarn':
        conf = SparkConf().setAppName("FM embedding").setMaster('yarn').set('spark.executor.memory', '10g')
    else:
        conf = SparkConf().setAppName("FM embedding")
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    print('********************** SparkContext and HiveContext is ready ************************')
    return sc, ss

if __name__ == '__main__':
    index_dic = get_new_index(ORIGIN_TRAIN)
    pickle.dump(index_dic, open(os.path.join(MODEL_DIR, 'index_dump'), 'wb'))
    _make_path(DATA_DIR)
    _make_path(MODEL_DIR)





