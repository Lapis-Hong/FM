import time
import cProfile
import re
import numpy as np
import pandas as pd
# from collections import deque
# from memory_profiler import profile

from conf import *
from prepareData import clock


@clock('Index mapping, Take {} sec.')
def get_new_index(infile):
    """
    if the original libsvm data index is not from zero, use this function
    index mapping 
    :return: {'66':0, '67':1, '75':2...}
    """
    with open(infile) as libsvm:
        index_set = set()
        for line in libsvm:
            line = line.strip('\n').split(' ')
            line.pop(0)
            for w in line:
                index = w.split(':')[0]
                if index not in index_set:
                    index_set.add(index)
        dic = {k: v for v, k in enumerate(index_set)}  # mapping the index to new index
        return dic


@clock('Successfully convert to the libfm format! Take {} sec.')
def convert_libfm(infile, outfile, isprd=False, reindex=False):
    """
    reindex from zero and remove the zero values pairs
    :param infile: the original libsvm format filename
    :param outfile: the libfm format filename
    :param isprd: flag
    :reindex: flag
    :return: 
    """
    if reindex:
        index_mapping = get_new_index(infile)
    with open(infile) as libsvm,\
            open(os.path.join(DATA_DIR, outfile), 'w') as libfm:
        for lineno, line in enumerate(libsvm):
            line = line.strip('\n').split(' ')  # convert string to list
            label = line.pop(0)  # string format '1' or '17040352933528'
            if isprd:  #
                libfm.write(label)
            else:
                libfm.write(label) if float(label) == 1 else libfm.write('-1')
            for element in line:
                temp = element.split(':')  # like ['5', '1']
                feature_value = temp[-1]  # string format
                feature_index_new = temp[0] if not reindex else index_mapping[temp[0]]
                if float(feature_value) != 0:  # remove zero feature index and values
                    libfm.write(' ' + str(feature_index_new) + ':' + feature_value)
            libfm.write('\n')
            if (lineno + 1) % 100000 == 0:
                print('Already convert %s samples ' % lineno)


@clock('Train test split completed, Take {} sec.')
def split_data(libfm_file, train_file, test_file, train_ratio=0.75):
    """
    make train and test file for libFM train 
    :param libfm_file: libfm_train_file
    :param train_file: libFM train data path
    :param test_file: libFM test data path
    :param train_ratio:
    :return: the libFM train and test data path
    """
    df = pd.read_csv(os.path.join(DATA_DIR, libfm_file), header=None)
    n, m = df.shape
    train = df.sample(frac=train_ratio)
    train_length = train.shape[0]
    train.to_csv(os.path.join(DATA_DIR, train_file), index=False, header=None)
    test_index = set(df.index) - set(train.index)
    test = df.iloc[list(test_index), :]
    test_length = test.shape[0]
    test.to_csv(os.path.join(DATA_DIR, test_file), index=False, header=None)
    print('Totol data length: {}, feature number: {}'.format(n, m))
    print('Train data length: {}'.format(train_length))
    print('Test data length:{}'.format(test_length))
    return os.path.abspath(train_file), os.path.abspath(test_file)


def feature_target_split(infile):
    """
    :param infile: original libfm train filename
    :return: target and feature 
    """
    with open(infile) as f:
        target = []
        feature = []
        index_mapping = get_new_index(infile)
        index_num = len(index_mapping)  # total feature number
        for line in f:
            line = line.strip('\n').split(' ')
            label = int(float(line.pop(0)))
            target.append(label)
            # dic = dict(map(lambda x: (float(x.split(':')[0]),float(x.split(':')[1])), line))
            dic = dict((float(w.split(':')[0]), float(w.split(':')[1])) for w in line)
            feature_line = (int(dic[i]) if i in dic.keys() else 0 for i in range(index_num))  # add 0 for the removed index
            feature.append(list(feature_line))
        return target, feature


def main():
    # print('the total input dataset has %d files' % len(LIBSVM_TRAIN_FILE))
    # for i in range(len(LIBSVM_TRAIN_FILE)):
    #     convert_libfm(LIBSVM_TRAIN_FILE[i], '{}({})'.format(LIBFM_TRAIN, i+1))
    # for i in range(len(LIBSVM_PRD_FILE)):
    #     convert_libfm(LIBSVM_PRD_FILE[i], LIBFM_PRD, isprd=True)
    # for f in [w for w in os.listdir('data') if re.findall(LIBFM_TRAIN, w)]:
    #     split_data(f, TRAIN, TEST)
    convert_libfm(LIBSVM_TRAIN_FILE, LIBFM_TRAIN)
    convert_libfm(LIBSVM_PRD_FILE, LIBFM_PRD, isprd=True)
    split_data(LIBFM_TRAIN, TRAIN, TEST)

if __name__ == '__main__':
    cProfile.run('main()')  # time analysis
