import sys
import os

import pandas as pd
import numpy as np
from pyspark import *
from pyspark.sql import SparkSession

from conf import *
from data_process import *

__all__ = ['make_path', 'split_data', 'split_data_fastfm', 'get_new_index',
           'get_target', 'target_feature_split', 'create_table', 'start_spark']


def make_path(*args):
    """make multiple path"""
    for dirname in args:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print('Already make directory: {0}'.format(dirname))


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
            try:
                dic = {k: v for v, k in enumerate(index_list)}  # mapping the index to new index
            except SyntaxError:
                pass
        return dic


def get_target(infile):
    with open(infile) as f:
        return [int(float(line.split(' ').pop(0))) for line in f]


@clock('Train test split has completed!')
def split_data(infile, train_file, test_file, train_ratio=TRAIN_RATIO, chunksize=200000, mode='error'):
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
            raise IOError('train file:{0} already exsits. Please change it in conf.py'.format(train_file))
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


@clock('Train test split for fastfm has completed!')
def split_data_fastfm(infile, train_ratio=TRAIN_RATIO):
    """fastFM package need the sparse matrix input for train the model"""
    from sklearn.model_selection import train_test_split
    from scipy.sparse import coo_matrix
    target, feature = target_feature_split(infile)
    target = np.array([1, -1, -1, -1, 1, 1, 1, -1, -1, 1])
    feature = coo_matrix(np.random.rand(10, 20))
    feature = coo_matrix(feature)
    target = np.array(target)
    train_X, test_X, train_y, test_y = train_test_split(feature, target,
                                                        train_size=train_ratio, random_state=0)
    return train_X, test_X, train_y, test_y


@clock('feature target split has finished!')
def target_feature_split(infile):
    """infile must have the same length"""
    target = []
    feature = []
    with open(infile) as f:
        for line in f:
            line = line.strip('\n').split(' ')
            target.append(2*float(line.pop(0))-1)
            feature_line = []
            for elem in line:
                ind, val = elem.split(':')
                feature_line.append(float(val))
            feature.append(feature_line)
    return np.array(target), np.array(feature)


def create_table(table, cols, types):
    """create table sql for hive"""
    head = 'create table if not exists ' + table + '( '
    body = []
    for i, j in zip(cols, types):
        body.append('{0} {1}'.format(i, j))
    body = ','.join(body)
    tail = ') partitioned by (dt string)'
    return head + body + tail


def start_spark(yarn_master=True):
    conf = SparkConf().setAppName("FM embedding").\
           set('spark.executor.memory', '10g').set('spark.driver.memory', '10g').\
           set('spark.driver.cores', '3')
    if yarn_master:
        conf.setMaster('yarn')
    else:
        conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    print('********************** SparkContext and HiveContext is ready ************************\n')
    return sc, ss







