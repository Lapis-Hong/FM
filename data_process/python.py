"""Python way to do the data preprocess, more flexible.  Default way
Notice: single thread is slow, but through concurrency 8x faster"""
import cProfile
import os
import glob
from concurrent import futures

from conf import *
from data_process import *
from data_process.util import *


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
    :param reindex: flag, True for reindex
    :param keep_zero: flag, True for not remove zero
    :return: None
    """
    if reindex:
        index_mapping = pickle.load(open(os.path.join(MODEL_DIR, 'index_dump'), 'rb'))
        print('index mapping has loaded...')
    else:
        index_mapping = None
    with open(infile) as libsvm:  # in python2.6, can not use with open...as..., open...as...
        with open(os.path.join(DATA_DIR, outfile), 'a+') as libfm:  # write and read, append
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


def multiprocess():
    """concurrent way -- multiprocess, much faster"""
    split(ORIGIN_TRAIN)
    split(ORIGIN_PRD, isprd=True)
    files = glob.glob('temp/train-part*')
    remove(os.path.join(DATA_DIR, FM_TRAIN))
    t0 = time.time()
    with futures.ProcessPoolExecutor() as executor:
        for f in files:
            executor.submit(convert_from_local, f, FM_TRAIN)
            # for func in executor.map(convert_from_local, files, [FM_TRAIN]*len(files)):
            #     func
    t1 = time.time()
    print('Total convert time: {0}'.format(t1 - t0))


def main():
    """do the data process in a pipeline"""
    load_data()
    multiprocess()
    split_data(FM_TRAIN, TRAIN, TEST)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')  # time analysis

