import time
import os
import shutil
data_libsvm_file = 'fmtrain20170403.txt'
data_temp_file = 'fmtrain20170403.temp2'
data_libfm_file = 'fmtrain20170403.libfm'

TRAIN_FILE = 'train'
TEST_FILE = 'test'
DATA_DIR = 'data'


def gen_path(directory, filename):
    return os.path.join(directory, filename)


# if exists dir, remove it, os.rmdir can only remove empty dir
def make_path(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def convert_libfm(input_file, temp_file, output_file):
    """
    :param input_file: the original libsvm format file
    :param temp_file: reindex from zero
    :param output_file: the libfm format filepath
    :return: 
    """
    time_start = time.time()
    with open(input_file) as data_libsvm,\
            open(os.path.join(DATA_DIR, temp_file), 'w') as data_temp,\
            open(os.path.join(DATA_DIR, output_file), 'w') as data_libfm:
        data_temp_list = []
        data_libfm_list = []
        count = 0
        for line in data_libsvm:
            line = line.strip('\n').split(' ')  # convert string to list
            label = line.pop(0)
            data_temp_list.append(str(label))
            data_libfm_list.append(str(label))

            index_all = [int(w.split(':')[0]) for w in line]  # save all the original index values
            index_set = list(set(index_all))  # unique original index values

            for element in line:
                temp = element.split(':')  # like ['5', '1']
                feature_value = temp[-1]  # string format
                feature_index_new = index_set.index(int(temp[0]))  # reindex from zero
                data_temp_list.append(' ' + str(feature_index_new) + ':' + str(feature_value))
                if feature_value != 0:  # remove zero feature index and values
                    data_libfm_list.append(' ' + str(feature_index_new) + ':' + str(feature_value))
            data_temp_list.append('\n')
            data_libfm_list.append('\n')
            count += 1
            if count % 10000 == 0:
                print(count)
        data_temp.write(''.join(data_temp_list))
        data_libfm.write(''.join(data_libfm_list))
        time_end = time.time()
        print('Successfully convert to the libfm format! Take %s sec:' % (time_end - time_start))


def split_data(file_path, train_file, test_file, k=5):
    """
    :param file_path: 
    :param train_file: 
    :param test_file: 
    :param k: 
    :return: 
    """
    with open(os.path.join(DATA_DIR, train_file), 'w') as fm_train, \
            open(os.path.join(DATA_DIR, test_file), 'w') as fm_test, \
            open(os.path.join(DATA_DIR, file_path)) as train_data:
        line = train_data.readline()
        count = 1
        while line:
            if count % k == 0:  # chose 20% as test data
                fm_test.write(line)
            else:
                fm_train.write(line)
            line = train_data.readline()
            count += 1
    return os.path.abspath(train_file), os.path.abspath(test_file)


if __name__ == '__main__':
    make_path(DATA_DIR)
    convert_libfm(data_libsvm_file, data_temp_file, data_libfm_file)
    split_data(data_libfm_file, TRAIN_FILE, TEST_FILE)
