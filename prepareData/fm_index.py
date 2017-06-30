import time
from conf import *


def convert_libfm(input_file, output_file):
    """
    reindex from zero and remove the zero values pairs
    :param input_file: the original libsvm format file
    :param output_file: the libfm format filepath
    :return: 
    """
    time_start = time.time()
    with open(input_file) as data_libsvm,\
            open(os.path.join(DATA_DIR, output_file), 'w') as data_libfm:
        data_libfm_list = []
        count = 0
        for line in data_libsvm:
            line = line.strip('\n').split(' ')  # convert string to list
            label = line.pop(0)
            data_libfm_list.append(str(label))

            index_all = [int(w.split(':')[0]) for w in line]  # save all the original index values
            index_set = list(set(index_all))  # unique original index values

            for element in line:
                temp = element.split(':')  # like ['5', '1']
                feature_value = temp[-1]  # string format
                feature_index_new = index_set.index(int(temp[0]))  # reindex from zero
                if feature_value != 0:  # remove zero feature index and values
                    data_libfm_list.append(' ' + str(feature_index_new) + ':' + str(feature_value))
            data_libfm_list.append('\n')
            count += 1
            if count % 10000 == 0:
                print(count)
        data_libfm.write(''.join(data_libfm_list))
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
    return os.path.abspath(train_file), os.path.abspath(test_file)


def main():
    # for i in range (len(LIBSVM_TRAIN_FILE)):
    #     convert_libfm(LIBSVM_TRAIN_FILE[i], LIBFM_TRAIN_FILE+'%s' % (i+1))

    # for i in range(len(LIBSVM_PRD_FILE)):
    #     convert_libfm(LIBFM_PRD_FILE, LIBFM_PRD_FILE+'%s' % (i+1))

    convert_libfm(LIBSVM_TRAIN_FILE, LIBFM_TRAIN_FILE)
    convert_libfm(LIBSVM_PRD_FILE, LIBFM_PRD_FILE)
    # trainData, testData = map(split_data, ORIGINAL_DATA_PATH, TRAIN_PATH*len(), TEST_PATH)
    for i in range(len(LIBFM_TRAIN_FILE)):
        split_data(LIBFM_TRAIN_FILE[i], TRAIN_FILE, TEST_FILE)
        print('%s split completed, writing train data into %s' % (LIBFM_TRAIN_FILE[i], TRAIN_FILE))


if __name__ == '__main__':
    main()
