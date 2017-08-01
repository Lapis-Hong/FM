try:
    import cPickle as pickle
except:
    import pickle
import cProfile
# from collections import deque
# from memory_profiler import profile
from conf import *
from data_process import clock
from data_process.util import split_data


def reformat(isprd, reindex, keep_zero, index_mapping):
    """use closure to pass the additional parameters"""
    def inner(each_line):
        """temp is each line, string format"""
        newline = []
        line = each_line.strip('\n').split(' ')  # convert string to list
        label = line.pop(0)  # string format '1' or '17040352933528'
        newline.append(label) if isprd else newline.append(str(int(2*float(label)-1)))  # label transform
        for elem in line:
            index, value = elem.split(':')  # like ['5', '1'], string format
            if keep_zero:
                index_new = index if not reindex else index_mapping[int(index)]
                newline.append(' ' + str(index_new) + ':' + value)
            elif float(value) != 0:  # remove zero feature index and values
                index_new = index if not reindex else index_mapping[int(index)]
                newline.append(' ' + str(index_new) + ':' + value)
        return ' '.join(newline)
    return inner


@clock('Successfully convert to the libfm format!')
def convert_from_local(infile, outfile, isprd=False, reindex=False, keep_zero=False):
    """
    reindex from zero and remove the zero values pairs
    :param infile: the original libsvm format filename
    :param outfile: the libfm format filename
    :param isprd: flag
    :reindex: flag
    :return: 
    """
    #global index_mapping
    if reindex:
        index_mapping = pickle.load(open('dump', 'rb'))
        print('index mapping has dumped')
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
                print('Already convert %s samples ' % (lineno + 1))

# def feature_target_split(infile):
#     """
#     :param infile: original libfm train filename
#     :return: target and feature
#     """
#     with open(infile) as f:
#         target = []
#         feature = []
#         index_mapping = get_new_index(infile)
#         index_num = len(index_mapping)  # total feature number
#         with open('target', 'wb') as target,\
#             open('feature', 'wb') as feature:
#             for line in f:
#                 line = line.strip('\n').split(' ')
#                 label = int(float(line.pop(0)))
#                 pickle.dump(label, target)
#                 # dic = dict(map(lambda x: (float(x.split(':')[0]),float(x.split(':')[1])), line))
#                 dic = dict((float(w.split(':')[0]), float(w.split(':')[1])) for w in line)
#                 feature_line = [int(dic[i]) if i in dic.keys() else 0 for i in range(index_num)]  # add 0 for the removed index
#                 #pickle.dump(feature_line, feature)
#                 feature.append(list(feature_line))
#         #return target, feature

if __name__ == '__main__':
    #index_mapping = pickle.load(open('dump', 'rb'))
    convert_from_local(ORIGIN_TRAIN, FM_TRAIN, reindex=True)
    # convert_fromLocal(ORIGIN_TRAIN, FM_TRAIN+'keep_zero', reindex=True, keep_zero=True)
    # convert_fromLocal(LIBSVM_PRD_FILE, LIBFM_PRD, isprd=True)
    split_data(FM_TRAIN, TRAIN, TEST)
    # cProfile.run('main()')  # time analysis

