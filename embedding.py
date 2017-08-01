from data_process import clock
from conf import *


def embedding_index_init(total):
    return set(range(1, total+1)) - not_embedding_index()


def not_embedding_index():
    return set(range(1, 565))


@clock('Successfully saved embedding data into %s.' % EMBEDDING_PATH)
def embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=EMBEDDING_PATH, threshold=564, isappend=True, islibsvm=False):
    """
    take the fm latent vectors as the embedding features, generate new data
    :param infile: original data path, the original data has must have the libsvm format
    :param outfile: embedding data path
    :param latent: latent vector
    :param threshold: the threshold for continuous and category features
    :param isappend: True: add embedding category features to the original continuous features, 
                     False: for only generate the embedding category features
    :param islibsvm: True for saving the libsvm format, False for saving the dataframe format
    """
    print('*'*30 + 'Start saving the embedding data!' + '*'*30)
    index_mapping = pickle.load(open('dump', 'rb'))
    print('index mapping:{}'.format(index_mapping))
    with open(infile) as f1, open(outfile, 'w') as f2:
        total_dim, not_embedding_dim, embedding_dim = set(), set(), set()
        for lineno, line in enumerate(f1):
            temp = line.strip('\n').split(' ')  # ['1:0','2:1','5:0']
            f2.write(temp.pop(0))  # write order_id to the first col, string
            feature_line = []
            cnt1, cnt2 = 0, 0
            for item in temp:  # item: '1:0'
                index, value = item.split(':')  # string
                if int(index) <= threshold:  # write the continuous feature
                    if isappend:
                        cnt1 += 1
                        f2.write(' '+value) if not islibsvm else f2.write(' '+item)
                else:
                    if float(value) == 1:  # onehot index, value one to activate embedding
                        cnt2 += 1
                        index_new = index_mapping[int(index)]
                        feature_line.extend(latent[index_new])
            embedding_index = range(threshold + 1, threshold + cnt2 * len(latent[0]) + 1)
            total_cnt = cnt1 + cnt2 * len(latent[0])
            assert len(feature_line) == len(embedding_index)  # embedding feature length must same with embedding dim
            if islibsvm:
                pairs = (str(ind)+':'+str(val) for ind, val in zip(embedding_index, feature_line))
                f2.write(' '+' '.join(pairs) + '\n')
            else:
                val = (str(val) for val in feature_line)
                f2.write(' '+' '.join(val) + '\n')
            if (lineno+1) % 10 == 0:
                print('Already write {0} embedding data'.format(lineno+1, outfile))

            if cnt1 not in not_embedding_dim:
                not_embedding_dim.add(cnt1)
            if cnt2 not in embedding_dim:
                embedding_dim.add(cnt2)
            if total_cnt not in total_dim:
                total_dim.add(total_cnt)
        if len(embedding_dim) != 1:
            print('Warning: the embedding feature length is different!')
            print('embedding feature dim set:{0}'.format(sorted(list(embedding_dim))))
        if len(not_embedding_dim) != 1:
            print('Warning: the not embedding feature length is different!')
            print('not embedding feature dim set:{0}'.format(sorted(list(not_embedding_dim))))
        if len(total_dim) != 1:
            print('Warning: total data length is different!')
            print('total dim set:{0}'.format(sorted(list(total_dim))))

        print('not embedding feature length is {0}'.format(max(not_embedding_dim)))
        print('embedding feature length is {0}'.format(max(embedding_dim)*len(latent[0])))
        print('total feature length is {0}'.format(max(total_dim)))
        print('total data length is {0}'.format(lineno))


@clock('Embedding data generator is ready')
def embedding_generator(latent, infile=ORIGIN_TRAIN, isappend=True, threshold=564):
    """only support the infile with same feature length"""
    index_mapping = pickle.load(open('dump', 'rb'))
    with open(infile) as f:
        for line in f:
            feature_line = []
            temp = line.strip('\n').split(' ')  # ['1:0','2:1','5:0']
            label = temp.pop(0)
            feature_line.append(label)
            for item in temp:
                index, value = item.split(':')
                if int(index) <= threshold:
                    if isappend:
                        feature_line.append(float(value))
                else:
                    if float(value) == 1:  # onehot index, value one to activate embedding
                        index_new = index_mapping[int(index)]
                        feature_line.extend(latent[index_new])
            yield feature_line






