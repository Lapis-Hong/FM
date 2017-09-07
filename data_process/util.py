from __future__ import print_function
import os
import subprocess

import pandas as pd
import numpy as np
from pyspark import *
from pyspark.sql import SparkSession

from conf import *
from data_process import *

__all__ = ['make_path', 'remove', 'hdfs_to_local', 'local_to_hdfs', 'local_to_hive',
           'split', 'split_data', 'split_data_fastfm', 'get_new_index',
           'get_target', 'target_feature_split', 'gen_libsvm', 'load_data',
           'create_table', 'start_spark', 'pandas_to_spark']


def make_path(*args):
    """make multiple paths, equivalent to linux mkdir command"""
    for dirname in args:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print('Already make directory: {0}'.format(dirname))


def remove(*args):
    """remove multiple files equivalent to linux rm command"""
    for filename in args:
        if os.path.exists(filename):
            os.remove(filename)
            print('Already remove file: {0}'.format(filename))


def hdfs_to_local(hdfs_path, file_name):
    """move hdfs datasets to current dir"""
    try:
        subprocess.check_call("hadoop fs -text {0}/* > {1}".format(hdfs_path, file_name), shell=True)
        print('Already load original data {0} from {1}!'.format(file_name, hdfs_path))
    except subprocess.CalledProcessError as e:
        print("Command Error:", end=' ')
        print(e)


def local_to_hdfs(file_name, hdfs_path, keep_local=True):
    """move current dir datasets to hdfs"""
    subprocess.call("hadoop fs -mkdir -p {0}".format(hdfs_path), shell=True)
    subprocess.call("hadoop fs -put {0} {1}".format(file_name, hdfs_path), shell=True)
    print('Already move {0} to {1}'.format(file_name, hdfs_path))
    if not keep_local:
        remove(file_name)


def local_to_hive(file_name, table, keep_local=True):
    """write local data to hive"""
    subprocess.call("hive -e 'load data local inpath \"{0}\" into table {1}'"
                    .format(file_name, table), shell=True)
    if not keep_local:
        remove(file_name)
        print('Remove local file {0}'.format(file_name))


@clock('Successfully generate the libsvm format!')
def gen_libsvm(infile, outfile):
    awk_command = 'awk \'{printf $1} {for(i=2; i<=NF; i++) {printf "  "i-1":"$i}} {print " "}\' '
    cmd = awk_command + infile + ">" + outfile
    print('The shell command is:{0}'.format(cmd))
    subprocess.check_call(cmd, shell=True)


@clock('Already split original data into temp/')
def split(infile, isprd=False):
    try:
        os.mkdir('temp')
    except OSError:
        pass
    if isprd:
        cmd = 'split -a 3 -d -l 200000 {0} temp/prd-part'.format(infile)
    else:
        cmd = 'split -a 3 -d -l 200000 {0} temp/train-part'.format(infile)
    print('Running shell command:{0}'.format(cmd))
    subprocess.check_call(cmd, shell=True)


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
        dic = dict([(k, v) for v, k in enumerate(index_list)])  # mapping the index to new index
        return dic  # py2.7 can use OrderedDict


@clock('Already load train and prd dataset!')
def load_data():
    """make path, load dataset, dump index mapping and split data"""
    make_path(DATA_DIR, MODEL_DIR, EMBEDDING_DIR)
    hdfs_to_local(FROM_HDFS_TRAIN, ORIGIN_TRAIN)
    hdfs_to_local(FROM_HDFS_PRD, ORIGIN_PRD)
    index_dic = get_new_index(ORIGIN_TRAIN)
    pickle.dump(index_dic, open(os.path.join(MODEL_DIR, 'index_dump'), 'wb'))
    split(ORIGIN_TRAIN)
    split(ORIGIN_PRD, isprd=True)


def get_target(infile):
    with open(infile) as f:
        return [int(float(line.split(' ').pop(0))) for line in f]


@clock('Train test split has completed!')
def split_data(infile, train_file, test_file, train_ratio=TRAIN_RATIO, chunksize=200000, mode='overwrite'):
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


@clock('Train test split for fastfm has completed!')
def split_data_fastfm(infile, train_ratio=TRAIN_RATIO):
    """fastFM package need the sparse matrix input for train the model"""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        from sklearn.cross_validation import train_test_split
    from scipy.sparse import coo_matrix
    target, feature = target_feature_split(infile)
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
    tail = ')'  # partitioned by (dt string)'
    return head + body + tail


def start_spark(yarn_master=True):
    conf = SparkConf().setAppName("FM embedding").\
           set('spark.executor.memory', EXECUTOR_MEMORY).set('spark.driver.memory', DRIVER_MEMORY).\
           set('spark.driver.cores', DRIVER_CORES).set('spark.cores.max', CORES_MAX)
    if yarn_master:
        conf.setMaster('yarn')
    else:
        conf.setMaster('local[*]')
    try:
        sc.stop()
    except:
        pass
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # why not work ??
    ss = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    print('*'*30 + 'SparkContext and HiveContext is ready' + '*'*30 + '\n')
    return sc, ss


def pandas_to_spark(file_name):
    # df = spark.read.format("csv").options(header="true").load("fmtrain20170704")
    # df.dropna(axis=1, how='all')
    # df[df==0] = np.nan  # can take 0 to NaN
    # pandas_df.info()
    _, ss = start_spark()
    df = pd.read_csv(file_name)  # null is NaN in pandas
    if df.isnull().any().sum() > 0:
        category_cols = df.columns[df.dtypes == object]  # object is mix type
        if len(category_cols) > 0:
            df.fillna('0', inplace=True)
            numerical_cols = df.columns[df.dtypes != object]
            df = df[numerical_cols].astype(float)  # change the string '0' to float 0.0
            spark_df = ss.createDataFrame(df)
        else:
            spark_df = ss.createDataFrame(df)
    else:
        spark_df = ss.createDataFrame(df)
    return spark_df


