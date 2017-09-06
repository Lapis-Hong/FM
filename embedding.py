"""Contains four functions to do the embedding, write embedding data to hive and hdfs"""
import os
import glob
import subprocess
from concurrent import futures

from pyspark import *
from pyspark.sql.types import *

from conf import *
from train import train
from data_process import *
from data_process.util import *

sc, ss = start_spark()


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
    ss.sql(sql)
    print('Already create table <{0}> in hive'.format(LATENT_TABLE))
    df = ss.createDataFrame(latent)
    df.registerTempTable('dataset')
    ss.sql("insert into table {0} select * from dataset".format(LATENT_TABLE))
    print('Successfully write latent vector to hive!')


@clock('Successfully write embedding data to local')
def embedding_to_local(latent, infile, outfile, threshold=THRESHOLD,
                       isappend=ISAPPEND, islibsvm=True, isprd=False, keep_local=True):

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
                    print('Already write {0} embedding data into {1}'.format(lineno+1, outfile))

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
    if isprd:
        local_to_hdfs(outfile, TO_HDFS_PRD, keep_local)
    else:
        local_to_hdfs(outfile, TO_HDFS_TRAIN, keep_local)


def embedding_generator(latent, infile, hastarget=False, threshold=THRESHOLD, isappend=ISAPPEND, islibsvm=KEEP_LOCAL):
    """
    Most important function, only support infile with same feature length
    :param latent: latent matrix, list
    :param infile: train or prd file
    :param hastarget: when infile=ORIGIN_PRD, set True to add the 'target' field for embedding into hive
    :param threshold: only index after threshold will do embedding
    :param isappend: True for including the original feature, False for only the embedding feature
    :param islibsvm: True for libsvm format
    :return: a generator list format
    """
    index_mapping = pickle.load(open(os.path.join(MODEL_DIR, 'index_dump'), 'rb'))
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
                yield ' '.join(target + feature_line)  # string '1 0:1 1:1 2:0' sc.parallelize much faster than a list
            else:
                yield target + feature_line  # list ['1', '0.2', '0', '0.2342']
        print('{0} data generator is ready...'.format(infile))


@clock('Successfully write all embedding data to hive!')
def embedding_to_hive(generator, batch_size=BATCH_SIZE):
    line = generator.next()
    dim = len(line)
    print('The first row of embedding data: {0}'.format(line))
    print('Total embedding dim is %s' % dim)
    schema = StructType([StructField("order_id", StringType(), True),
                         StructField("target", ByteType(), True)])
    for i in range(dim - 2):
        schema.add("f %d" % i, StringType(), True)
    cols = ['order_id'] + ['target'] + ['f{0}'.format(i) for i in range(dim - 2)]
    types = ['string'] * dim
    sql = create_table(EMBEDDING_TABLE, cols, types)  # schema type and table type different??
    ss.sql(sql)  # create embedding table in hive
    print('Successfully create table {0} in hive!'.format(EMBEDDING_TABLE))

    batch_no = 1
    while 1:
        df = (generator.next() for i in range(batch_size))
        try:
            df.next()  # but loss this sample
        except StopIteration:
            break
        t0 = time.time()
        rdd = sc.parallelize(df, 24)  # slow take 80s per 10000 data
        t1 = time.time()
        # print('the first rdd {0}'.format(rdd.first()))
        embedding_df = ss.createDataFrame(rdd, schema)  # fast 1s
        # print('the first embedding feature {0}'.format(embedding_df.first()))
        # embedding_df.write.mode('append').saveAsTable(EMBEDDING_TABLE)
        embedding_df.registerTempTable("temp")
        ss.sql("insert into table {0} select * from temp".format(EMBEDDING_TABLE))  # slow 30s
        t2 = time.time()
        print('write {0} embedding data into table {1}  parallelize() take{2}; insert take{3}'
              .format(batch_size*batch_no, EMBEDDING_TABLE, t1-t0, t2-t1))
        batch_no += 1


def embedding_tolocal_tohive(generator, batch_size=BATCH_SIZE, keep_local=KEEP_LOCAL):
    """save embedding data to local partitions and write hive through local data"""
    line = generator.next()
    dim = len(line)
    print('The first row of embedding data: {0}'.format(line))
    print('Total embedding dim is %s' % dim)
    schema = StructType([StructField("order_id", StringType(), True),
                         StructField("target", ByteType(), True)])
    for i in range(dim - 2):
        schema.add("f %d" % i, StringType(), True)
    cols = ['order_id'] + ['target'] + ['f{0}'.format(i) for i in range(dim - 2)]
    types = ['string'] * dim
    sql = create_table(EMBEDDING_TABLE, cols, types)  # schema type and table type different??
    ss.sql(sql)  # create embedding table in hive
    print('Successfully create table {0} in hive!'.format(EMBEDDING_TABLE))

    batch_no = 1
    while 1:
        t0 = time.time()
        outpath = EMBEDDING_DIR + '/train_prd_part{0}'.format(batch_no)
        with open(outpath, 'w') as f:
            for i in range(batch_size):
                try:
                    line = generator.next()
                    f.write(line)
                except StopIteration:
                    break
        batch_no += 1
        t1 = time.time()
        local_to_hive(outpath, EMBEDDING_TABLE, keep_local)
        t2 = time.time()
        print('write {0} data into table {1}. to local take {2}, to hive take {3}'
              .format(batch_no*batch_size, EMBEDDING_TABLE, t1-t0, t2-t1))


@clock('Successfully write all embedding data to local, using generator!')
def embedding_to_local_generator(generator, outfile):
    with open(outfile, 'a+') as f:
        for lineno, line in enumerate(generator):
            f.write(' '.join(line) + '\n')
            if lineno == 0:
                print('The first row of embedding data: {0}'.format(line))
            if (lineno + 1) % 100000 == 0:
                print('Already write {0} samples into {1}'.format(lineno+1, outfile))


@clock('Successfully write all embedding data to hdfs!')
def embedding_to_hdfs(generator, hdfs_file, batch_size=BATCH_SIZE):
    """using spark directly write into hdfs, but very slow
    IO dense, no need much CPU, but more numslices can handle bigger batch_size"""
    batch_no = 1
    while 1:
        # a subgenertor much faster and save memory, list.append() is too slow
        df = (generator.next() for i in range(batch_size))
        try:
            df.next()
        except StopIteration:
            break
        hdfs_path = os.path.join(hdfs_file, str(batch_no))
        t0 = time.time()
        rdd = sc.parallelize(df, 24)  # default 2 numslices; take 36s each 10000 data
        t1 = time.time()
        rdd.saveAsTextFile(hdfs_path)  # take 6s
        t2 = time.time()
        print('write {0} embedding data into {1} sc.parallelize() take {2}  saveAsTextFile() take {3}'
              .format(batch_no*batch_size, hdfs_path, t1-t0, t2-t1))
        batch_no += 1


def hdfs_pipeline():
    """embedding to hdfs, multiprocess way"""
    latent, latent_dim = load_latent()
    files = os.listdir(EMBEDDING_DIR)
    remove(files)
    train_files = glob.glob('temp/train-part*')
    prd_files = glob.glob('temp/prd-part*')
    t0 = time.time()
    with futures.ProcessPoolExecutor() as executor:  # max_workers defaults to the cpu numbers
        for ind, (f1, f2) in enumerate(zip(train_files, prd_files)):
            executor.submit(embedding_to_local, latent, f1, EMBEDDING_DIR+'/train-part{0}'.format(ind))
            executor.submit(embedding_to_local, latent, f2, EMBEDDING_DIR+'/prd-part{0}'.format(ind))
    t1 = time.time()
    print('Total time: {0}'.format(t1 - t0))


def hive_pipeline():
    """latent & embedding to hive"""
    latent, latent_dim = load_latent()
    latent_to_hive(latent, latent_dim)

    generator_local_hive = embedding_generator(latent, ORIGIN_PRD, hastarget=True, islibsvm=False)
    embedding_tolocal_tohive(generator_local_hive)
    generator_hive = embedding_generator(latent, ORIGIN_PRD, hastarget=True, islibsvm=False)
    embedding_to_hive(generator_hive)
    sc.stop()
    ss.stop()


def main(hive=WRITE_HIVE, hdfs=WRITE_HDFS):
    if hdfs:
        hdfs_pipeline()
    if hive:
        hive_pipeline()


if __name__ == '__main__':
    main()
    # train_path = os.path.join(MODEL_DIR, EMBEDDING_TRAIN)
    # prd_path = os.path.join(MODEL_DIR, EMBEDDING_PRD)
    # remove(train_path, prd_path)
    # train_files = glob.glob('temp/train-part*')
    # prd_files = glob.glob('temp/prd-part*')
    #
    # latent, latent_dim = load_latent()
    # latent_to_hive(latent, latent_dim)
    #
    # """embedding to local, multiprocess way"""
    # t0 = time.time()
    # with futures.ProcessPoolExecutor() as executor:  # max_workers defaults to the cpu numbers
    #     for f1, f2 in zip(train_files, prd_files):
    #         executor.submit(embedding_to_local, f1, train_path)
    #         executor.submit(embedding_to_local, f2, prd_path)
    # t1 = time.time()
    # print('Total time: {0}'.format(t1-t0))
    #
    # """embedding to local, single thread way"""
    # # embedding_to_local(latent, infile=ORIGIN_TRAIN, outfile=train_path)
    # # embedding_to_local(latent, infile=ORIGIN_PRD, outfile=prd_path)
    # save_data_to_hdfs(train_path, TO_HDFS_TRAIN)
    # save_data_to_hdfs(prd_path, TO_HDFS_PRD)
    #
    # # """embedding to local using generator, multiprocess way"""
    # # t0 = time.time()
    # # with futures.ProcessPoolExecutor() as executor:
    # #     for f1, f2 in zip(train_files, prd_files):
    # #         executor.submit(embedding_to_local_generator, embedding_generator(latent, f1), train_path)
    # #         executor.submit(embedding_to_local_generator, embedding_generator(latent, f2), prd_path)
    # # t1 = time.time()
    # # print('Total time: {0}'.format(t1 - t0))
    #
    # """embedding to local, single thread way"""
    # # train_generator = embedding_generator(latent, ORIGIN_TRAIN)
    # # prd_generator = embedding_generator(latent, ORIGIN_PRD)
    # # embedding_to_local_generator(train_generator, train_path)
    # # embedding_to_local_generator(prd_generator, prd_path)
    # save_data_to_hdfs(train_path, TO_HDFS_TRAIN)
    # save_data_to_hdfs(prd_path, TO_HDFS_PRD)
    #
    # # embedding to hdfs
    # train_generator = embedding_generator(latent, ORIGIN_TRAIN)
    # prd_generator = embedding_generator(latent, ORIGIN_PRD)
    # embedding_to_hdfs(train_generator, TO_HDFS_TRAIN)
    # embedding_to_hdfs(prd_generator, TO_HDFS_PRD)
    #
    # # embedding to hive
    # generator_hive = embedding_generator(latent, ORIGIN_PRD, hastarget=True, islibsvm=False)
    # embedding_to_hive(generator_hive)
    # sc.stop()
    # ss.stop()
    #
