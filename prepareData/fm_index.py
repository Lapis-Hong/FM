import time
from conf import *
# from collections import deque
import cProfile
import re
# from memory_profiler import profile


def get_new_index(infile):
    """
    index mapping
    :param infile: 
    :return: {66:0, 67:1, 75:2...}
    """
    with open(infile) as libsvm:
        index_all = []
        for line in libsvm:
            line = line.strip('\n').split(' ')
            line.pop(0)
            index_all.extend([int(w.split(':')[0]) for w in line])
        index_set = set(index_all)  # int values can keep order after set operation
        dic = {k:v for v, k in enumerate(index_set)}  # mapping the index to new index
        return dic


def convert_libfm(infile, outfile):
    """
    reindex from zero and remove the zero values pairs
    :param infile: the original libsvm format filename
    :param outfile: the libfm format filename
    :return: 
    """
    time_start = time.time()
    index_mapping = get_new_index(infile)
    with open(infile) as libsvm,\
            open(os.path.join(DATA_DIR, outfile), 'w') as libfm:
        for lineno, line in enumerate(libsvm):
            line = line.strip('\n').split(' ')  # convert string to list
            label = line.pop(0)  # string format '1' or '17040352933528'
            libfm.write(label)
            for element in line:
                temp = element.split(':')  # like ['5', '1']
                feature_value = temp[-1]  # string format
                feature_index_new = index_mapping[int(temp[0])]
                # feature_index_new = index_set.index(int(temp[0]))  # reindex from zero .index() method slow!
                if float(feature_value) != 0:  # remove zero feature index and values
                    libfm.write(' ' + str(feature_index_new) + ':' + feature_value)
            libfm.write('\n')
            if (lineno+1) % 10000 == 0:
                print('Already convert %s samples ' % lineno)
        time_end = time.time()
        print('Successfully convert to the libfm format! Take %s sec:' % (time_end - time_start))


def split_data(libfm_file, train_file, test_file, k=5):
    """
    make train and test file for libFM train 
    :param libfm_file: libfm_train_file
    :param train_file: libFM train data path
    :param test_file: libFM test data path
    :param k: 1/k is the test percentage
    :return: the libFM train and test data path
    """
    start = time.time()
    with open(os.path.join(DATA_DIR, train_file), 'w') as fm_train, \
            open(os.path.join(DATA_DIR, test_file), 'w') as fm_test, \
            open(os.path.join(DATA_DIR, libfm_file)) as f:
        line = f.readline()
        count = 1
        while line:
            if count % k == 0:  # chose 20% as test data
                fm_test.write(line)
            else:
                fm_train.write(line)
            line = f.readline()
            count += 1
    end = time.time()
    print('{} split completed, writing train data into {}, test data into {}. Take {} sec:'
          .format(libfm_file, train_file, test_file, end-start))
    return os.path.abspath(train_file), os.path.abspath(test_file)


def feature_target_split(test_file):
    """
    :param test_file: 
    :return: target and feature 
    """
    with open(test_file) as f:
        target = []
        feature = []
        for line in f:
            temp = line.strip('\n').split(' ')
            target.append(float(temp.pop(0)))
            feature.append([float(w.split(':')[1]) for w in temp])
        return target, feature


def main():
    print('the total input dataset has %d files' % len(LIBSVM_TRAIN_FILE))
    for i in range (len(LIBSVM_TRAIN_FILE)):
        convert_libfm(LIBSVM_TRAIN_FILE[i], '{}({})'.format(LIBFM_TRAIN, i+1))

    for i in range(len(LIBSVM_PRD_FILE)):
        convert_libfm(LIBSVM_PRD_FILE[i], LIBFM_PRD)

    for f in [w for w in os.listdir('data') if re.findall(LIBFM_TRAIN, w)]:
        split_data(f, TRAIN, TEST)

if __name__ == '__main__':
    cProfile.run('main()')  # time analysis
