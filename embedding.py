"""Contains four functions to do the embedding"""
import gc
import os

from pyspark import *
from pyspark.sql.types import *

from conf import *
from train import train
from data_process import *
from data_process.shell import save_data_to_hdfs
from data_process.util import *


def load_latent():
    """load latent vector"""
    try:
        latent = pickle.load(open(os.path.join(MODEL_DIR, 'latent_dump.libfm'), 'rb'))
    except IOError:
        model_default = train(silent=False)
        latent = model_default.pairwise_interactions

    print('The first latent row: {0}'.format(latent[0]))
    dim = len(latent[0])
    print('latent vector dim is %s' % dim)
    return latent, dim


def latent_to_hive(latent, dim):
    cols = ['dim{0}'.format(i+1) for i in range(dim)]
    types = ['string'] * dim
    sql = create_table(LATENT_TABLE, cols, types)
    global ss
    ss.sql(sql)
    print('Already create {0} in hive'.format(LATENT_TABLE))
    df = ss.createDataFrame(latent)
    df.registerTempTable('dataset')
    ss.sql("insert into table {0} select * from dataset".format(LATENT_TABLE))
    print('Successfully write latent vector to hive!')


@clock('Successfully saved embedding data.')
def embedding_to_local(latent, infile, outfile, threshold=THRESHOLD, isappend=ISAPPEND, islibsvm=False):

    """
    take the fm latent vectors as the embedding features, generate new data
    :param latent: latent vector
    :param infile: original train or prd data path
    :param outfile: embedding train or prd data path
    :param threshold: the threshold for continuous and category features
    :param isappend: True: add embedding category features to the original continuous features, 
                     False: for only generate the embedding category features
    :param islibsvm: True for saving the libsvm format, False for saving the dataframe format
    """
    print('*'*30 + 'Start saving the embedding data...' + '*'*30)
    index_mapping = pickle.load(open(os.path.join(MODEL_DIR, 'index_dump'), 'rb'))
    # print('index mapping:{0}'.format(index_mapping))
    with open(infile) as f1:
        with open(outfile, 'w') as f2:
            total_dim, not_embedding_dim, embedding_dim = set(), set(), set()
            for lineno, line in enumerate(f1):
                temp = line.strip('\n').split(' ')  # ['1:0','2:1','5:0']
                f2.write(temp.pop(0))  # write order_id to the first col, string
                feature_line = []
                cnt1, cnt2 = 0, 0  # record the not_embedding_dim and the embedding dim
                for item in temp:  # item: '1:0'
                    index, value = item.split(':')  # string
                    if int(index) <= threshold:  # write the continuous feature
                        if isappend:
                            cnt1 += 1
                            f2.write(' '+item) if islibsvm else f2.write(' '+value)
                    else:
                        if float(value) == 1:  # onehot index, value one to activate embedding
                            cnt2 += 1
                            index_new = index_mapping[int(index)]
                            feature_line.extend(latent[index_new])
                embedding_index = range(threshold + 1, threshold + cnt2 * len(latent[0]) + 1)
                total_cnt = cnt1 + cnt2 * len(latent[0])
                assert len(feature_line) == len(embedding_index)  # embedding feature length must same with embedding dim
                if islibsvm:
                    each_line = (str(ind) + ':' + str(val) for ind, val in zip(embedding_index, feature_line))
                else:
                    each_line = (str(val) for val in feature_line)
                f2.write(' ' + ' '.join(each_line) + '\n')

                if (lineno+1) % 100000 == 0:
                    print('Already write {0} embedding data to {1}'.format(lineno+1, outfile))

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

            print('not embedding feature length is {0}'.format(max(not_embedding_dim)))
            print('embedding feature length is {0}'.format(max(embedding_dim)*len(latent[0])))
            print('total feature length is {0}'.format(max(total_dim)))
            print('total data length is {0}'.format(lineno))


@clock('Embedding data generator is ready...')
def embedding_generator(latent, infile, hastarget=False, threshold=THRESHOLD, isappend=ISAPPEND, islibsvm=True):
    """
    only support infile with same feature length
    :param latent: latent matrix, list
    :param infile: train or prd file
    :param hastarget: when infile=ORIGIN_PRD, set True to add the 'order_id' field for embedding into hive
    :param threshold: only index after threshold will do embedding
    :param isappend: True for including the original feature, False for only the embedding feature
    :param islibsvm: True for libsvm format
    :return: a generator list format
    """
    index_mapping = pickle.load(open(os.path.join(DATA_DIR, 'index_dump'), 'rb'))
    total_dim = 0
    if hastarget:
        targets = get_target(ORIGIN_TRAIN)  # list
    with open(infile) as f:
        for lineno, line in enumerate(f):
            target = []
            feature_line = []
            temp = line.strip('\n').split(' ')  # ['1:0','2:1','5:0']
            target.append(temp.pop(0))  # add order_id
            if hastarget:
                target.append(targets[lineno])  # add target
            for item in temp:
                index, value = item.split(':')
                if int(index) <= threshold:
                    if isappend:
                        feature_line.append(value)  # string
                else:
                    if float(value) == 1:  # onehot index, value one to activate embedding
                        index_new = index_mapping[int(index)]
                        feature_line.extend((str(i) for i in latent[index_new]))  # float
            if total_dim == 0:
                total_dim = len(feature_line)
                print('total feature length is: {0}'.format(total_dim))
            if len(feature_line) != total_dim:
                raise ValueError('the feature length is different, expect to be the same!')
            if islibsvm:
                feature_line = [str(ind) + ':' + val for ind, val in enumerate(feature_line)]
            yield target + feature_line  # list format


@clock('Successfully write all embedding data to hive!')
def embedding_to_hive(generator, batch_size=BATCH_SIZE):
    line = generator.next()
    dim = len(line)
    print('The first row of embedding data: {0}'.format(line))
    print('Total embedding dim is %s' % dim)
    schema = StructType([StructField("order_id", StringType(), True),
                         StructField("target", ByteType(), True)])
    for i in range(dim - 1):
        schema.add("f %d" % i, StringType(), True)
    cols = ['order_id'] + ['f{0}'.format(i) for i in range(dim - 1)]
    types = ['string'] * dim
    sql = create_table(EMBEDDING_TABLE, cols, types)
    ss.sql(sql)  # create embedding table in hive
    print('Successfully create table {0} in hive!'.format(EMBEDDING_TABLE))

    batch_no = 1
    while 1:
        df = (generator.next() for i in range(batch_size))
        try:
            df.next()
        except StopIteration:
            break
        rdd = sc.parallelize(df)
        # print('the first rdd {0}'.format(rdd.first()))
        embedding_df = ss.createDataFrame(rdd, schema)
        # print('the first embedding feature {0}'.format(embedding_df.first()))
        embedding_df.registerTempTable("temp")
        ss.sql("insert into table {0} select * from temp".format(EMBEDDING_TABLE))
        print('Already write {0} embedding data to hive'.format(batch_size*batch_no))
        batch_no += 1


@clock('Successfully write all embedding data to local, using generator!')
def embedding_to_local_generator(generator, outfile):
    with open(outfile, 'w') as f:
        for lineno, line in enumerate(generator):
            f.write(' '.join(line) + '\n')
            if lineno == 0:
                print('The first row of embedding data: {0}'.format(line))
            if (lineno + 1) % 100000 == 0:
                print('Already write {0} samples into {1}'.format(lineno+1, outfile))


@clock('Successfully write all embedding data to hdfs!')
def embedding_to_hdfs(generator, hdfs_file, batch_size=BATCH_SIZE):
    batch_no = 1
    while 1:
        # a subgenertor much faster and save memory, list.append() is too slow
        df = (generator.next() for i in range(batch_size))
        try:
            df.next()
        except StopIteration:
            break
        hdfs_path = os.path.join(hdfs_file, str(batch_no))
        sc.parallelize(df).saveAsTextFile(hdfs_path)
        print('write {0} data into {1}'.format(batch_no*batch_size, hdfs_path))
        batch_no += 1


sc, ss = start_spark()

if __name__ == '__main__':
    latent, latent_dim = load_latent()
    latent_to_hive(latent, latent_dim)

    # embedding to local
    embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
    embedding_to_local(latent, infile=ORIGIN_PRD, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))
    save_data_to_hdfs(os.path.join(MODEL_DIR, EMBEDDING_TRAIN), TO_HDFS_TRAIN)
    save_data_to_hdfs(os.path.join(MODEL_DIR, EMBEDDING_PRD), TO_HDFS_PRD)

    # train_generator = embedding_generator(latent, infile=ORIGIN_TRAIN, islibsvm=True)
    # prd_generator = embedding_generator(latent, infile=ORIGIN_PRD, islibsvm=True)
    # embedding_to_local_generator(train_generator, outfile=os.path.join(MODEL_DIR, EMBEDDING_TRAIN))
    # embedding_to_local_generator(prd_generator, outfile=os.path.join(MODEL_DIR, EMBEDDING_PRD))
    # save_data_to_hdfs(os.path.join(MODEL_DIR, EMBEDDING_TRAIN), TO_HDFS_TRAIN)
    # save_data_to_hdfs(os.path.join(MODEL_DIR, EMBEDDING_PRD), TO_HDFS_PRD)

    # embedding to hdfs
    train_generator = embedding_generator(latent, infile=ORIGIN_TRAIN)
    prd_generator = embedding_generator(latent, infile=ORIGIN_PRD)
    embedding_to_hdfs(train_generator, hdfs_file=TO_HDFS_TRAIN)
    embedding_to_hdfs(prd_generator, hdfs_file=TO_HDFS_PRD)

    # embedding to hive
    generator_hive = embedding_generator(latent, infile=ORIGIN_PRD)
    embedding_to_hive(generator_hive)
    sc.stop()
    ss.stop()

