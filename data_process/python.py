"""Python way to do the data preprocess, more flexible but slower."""
import cProfile
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from conf import *
from data_process import clock
from data_process.util import split_data


def reformat(isprd, reindex, keep_zero, index_mapping):
    """use closure to pass the additional parameters"""
    def inner(each_line):
        """each line, string format"""
        newline = []
        line = each_line.strip('\n').split(' ')  # convert string to list
        label = line.pop(0)  # string format '1' or '17040352933528'
        newline.append(label) if isprd else newline.append(str(int(2*float(label)-1)))  # label transform
        for elem in line:
            index, value = elem.split(':')  # like ['5', '1'], string format
            if keep_zero:
                index_new = index if not reindex else index_mapping[int(index)]
                newline.append(str(index_new) + ':' + value)
            elif float(value) != 0:  # remove zero feature index and values
                index_new = index if not reindex else index_mapping[int(index)]
                newline.append(str(index_new) + ':' + value)
        return ' '.join(newline)
    return inner


@clock('Successfully convert to the libfm format!')
def convert_from_local(infile, outfile, isprd=False, reindex=False, keep_zero=False):
    """
    relabel, remove zero, reindex is optional
    :param infile: the original libsvm format filename
    :param outfile: the libfm format filename
    :param isprd: flag, True for prd file
    :reindex: flag, True for reindex
    :keep_zero: flag, True for not remove zero
    :return: None
    """
    index_mapping = pickle.load(open('dump', 'rb')) if reindex else None
    print('index mapping has loaded...')
    with open(infile) as libsvm,\
            open(os.path.join(DATA_DIR, outfile), 'w') as libfm:
        for lineno, line in enumerate(libsvm):
        #     line = line.strip('\n').split(' ')  # convert string to list
        #     label = line.pop(0)  # string format '1' or '17040352933528'
        #     if isprd:  #
        #         libfm.write(label)
        #     else:
        #         libfm.write(label) if float(label) == 1 else libfm.write('-1')
        #     for element in line:
        #         index, value = element.split(':')  # like ['5', '1'], string format
        #         if keep_zero:
        #             index_new = index if not reindex else index_mapping[int(index)]
        #             libfm.write(' ' + str(index_new) + ':' + value)
        #         elif float(value) != 0:  # remove zero feature index and values
        #             index_new = index if not reindex else index_mapping[int(index)]
        #             libfm.write(' ' + str(index_new) + ':' + value)
        #     libfm.write('\n')
            newline = reformat(isprd, reindex, keep_zero, index_mapping)(line)
            libfm.write(newline + '\n')
            if (lineno + 1) % 100000 == 0:
                print('Already convert %s samples' % (lineno + 1))


if __name__ == '__main__':
    convert_from_local(ORIGIN_TRAIN, FM_TRAIN)
    split_data(FM_TRAIN, TRAIN, TEST, mode='overwrite')
    # cProfile.run('main()')  # time analysis

