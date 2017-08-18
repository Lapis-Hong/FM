try:
    import cPickle as pickle
except:
    import pickle
import sys

import pandas as pd

from conf import *
from data_process import clock



@clock('Index mapping.')
def get_new_index(infile, foreachline=False):
    """
    if the original libsvm data index is not from zero, use this function
    index mapping 
    foreachLine: if the original libsvm index is different for each line, then set True
    :return: {66:0, 67:1, 75:2...}
    """
    with open(infile) as libsvm:
        index_set = set()  # save all the original index
        for line in libsvm:
            line = line.strip('\n').split(' ')
            line.pop(0)
            for w in line:
                index = int(w.split(':')[0])
                if index not in index_set:
                    index_set.add(index)
            if not foreachline:
                break
        index_list = list(index_set)
        index_list.sort()
        if sys.version < '2.7':
            dic = dict([(k, v) for v, k in enumerate(index_list)])
        else: dic = {k: v for v, k in enumerate(index_list)}  # mapping the index to new index
        return dic


def get_target(infile):
    with open(infile) as f:
        return [int(float(line.split(' ').pop(0))) for line in f]


@clock('Train test split completed!')
def split_data(libfm_file, train_file, test_file, train_ratio=0.8, mode='error'):
    """
    make train and test file for libFM train, shuffle the data
    :param libfm_file: FM_TRAIN_file
    :param train_file: libFM train data path
    :param test_file: libFM test data path
    :param train_ratio:
    :param mode: 
    :return: the libFM train and test data path
    """
    if os.path.exists(train_file):
        if mode == 'error':
            raise IOError('{} is already exsits. Please change it in conf.py'.format(train_file))
        elif mode == 'overwrite':
            os.remove(train_file)
            os.remove(test_file)
        elif mode == 'append':
            pass
        else:
            raise TypeError('No such mode')
    train_length, test_length = 0, 0
    for df in pd.read_csv(os.path.join(DATA_DIR, libfm_file), header=None, chunksize=200000):
        # n, m = df.shape
        train = df.sample(frac=train_ratio)
        train_length += train.shape[0]
        train.to_csv(os.path.join(DATA_DIR, train_file), mode='a', index=False, header=None)
        test_index = set(df.index) - set(train.index)
        test = df.iloc[list(test_index), :]
        test_length += test.shape[0]
        test.to_csv(os.path.join(DATA_DIR, test_file), mode='a', index=False, header=None)
        # print('Totol data length: {}, feature number: {}'.format(n, m))
    print('Train data length: {0}'.format(train_length))
    print('Test data length: {0}'.format(test_length))
    return os.path.abspath(train_file), os.path.abspath(test_file)


def gen_cmd_line(train_path, test_path):
    cmdline = []
    # essential
    cmdline.extend(["./libFM", "-task", "c", "-train", train_path,
                    "-test", test_path, "-dim", "1,1,8"])
    # parameter
    cmdline.extend(["-iter", "234", "-method", "sgd", "-learn_rate",
                    "0.01", "-regular", "'0,0,0.01'", "-init_stdev", "0.1"])
    # save model and predict
    cmdline.extend(["-save_model", "model", "-out", "output"])
    return cmdline


if __name__ == '__main__':
    index_dic = get_new_index(ORIGIN_TRAIN)
    pickle.dump(index_dic, open('dump', 'wb'))
    # index_mapping = pickle.load(open('dump', 'rb'))




