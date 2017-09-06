"""Read original data (before feature transform) from hive, do the feature transform directly using Spark

TODO: need to improve the performance and test for multiple days data
"""
import time
from collections import Counter

from pyspark import *
from pyspark.ml.feature import *

from conf import *
from mysql import trans_dic
from transformer import *
from data_process import clock
from data_process.util import start_spark
from data_process.shell import hdfs_to_local, gen_libsvm

# test_dic = {'age': 'discretize(20)', 'sex': 'onehot', 'wait_pay': 'bucket(min,10,100,max);onehot()'}


def read_from_hive(table, from_dt, to_dt):
    _, ss = start_spark()
    df = ss.sql('select * from {0} where dt between {1} and {2}'.format(table, from_dt, to_dt))
    print('Successfully read the data from hive, {0} rows, {1} cols'.format(df.count(), len(df.columns)))
    return df


def parse_transform(transform_dict):
    transform_list = []
    for field in transform_dict.keys():
        if transform_dict[field] is None:
            transform_list.append(('delete', field))
        else:
            transformer = transform_dict[field].split(';')
            for transform in transformer:
                if transform.startswith('discretize'):
                    start = transform.find('(') + 1
                    end = transform.find(')')
                    arg = transform[start:end]
                    transform_list.append(('discretizer', field, int(arg)))
                elif transform.startswith('bucket'):
                    start = transform.find('(') + 1
                    end = transform.find(')')
                    arg = transform[start:end]
                    arg = arg.replace('min', "-inf")
                    arg = arg.replace('max', "inf")
                    transform_list.append(('bucketizer', field, [float(w) for w in arg.split(',')]))
                elif transform.startswith('onehot'):
                    transform_list.append(('onehot', field))
                elif transform.startswith('standardnormalize'):
                    transform_list.append(('standard_scaler', field))
                elif transform.startswith('Untransform'):
                    pass
                else:
                    raise Exception('transform {0} of field {1} is not found, please check the transform'.format(transform, field))
    c = Counter(elem[0] for elem in transform_list)
    print(c)
    print('There are total {0} transformers need to be transform'.format(len(transform_list)))
    return transform_list  # each elem is a tuple


@clock()
def transform_pipeline(data, transform):
    # data_type = dict(data.dtypes)
    df = data.drop('order_sn', 'dt', 'user_id').cache()
    for index, item in enumerate(transform):
        tran, col = item[0], item[1]
        # change the column type, alias for df.na.fill()
        df = df.withColumn(col, df[col].astype('double')).fillna(0, col)
        t0 = time.time()
        if tran == 'bucketizer':
            df = bucktizer(df, col, 'temp', splits=item[2], drop=True)
        elif tran == 'discretizer':
            df = discretizer(df, col, 'temp', numBuckets=item[2], drop=True)
        elif tran == 'onehot':
            df = onehot(df, item[1], 'temp', drop=True)
        elif tran == 'standard_scaler':
            df_temp = vector_assembler(df, [col], 'Vector', drop=True)  # convert double to the Vectors type
            df = standard_scaler(df_temp, 'Vector', 'temp', drop=True)
        elif tran == 'delete':
            df = df.drop(col)
        else:
            raise TypeError('Unknown transfrom type, please check the input')
        df = df.withColumnRenamed('temp', col)#.cache()
        print('The {0}th feature {1}; transform {2} has finished, take {3} sec.'.format(index+1, col, tran, time.time()-t0))
    print('Feature transform has done!')
    print('Data after transform has {0} columns, as follows\n{1}'
          .format(len(df.columns), df.columns))
    return df.drop('order_id'), df.drop('target')


@clock()
def write_to_hdfs(data, path):
    data.write.mode('overwrite').save(path)  # default is 'error' mode 'ignore' 'append'
    print('Already save data to {0}'.format(path))


if __name__ == '__main__':
    train_temp_path = 'hdfs://bipcluster/user/u_jrd_lv1/fm_train'
    prd_temp_path = 'hdfs://bipcluster/user/u_jrd_lv1/fm_prd'
    # begin = datetime.date(2017, 4, 21)
    # end = datetime.date(2017, 7, 20)
    # for i in range((end-begin).days+1):
    #     day = begin + datetime.timedelta(days=i)
    #     dt = day.strftime("%Y%m%d")
    dataset = read_from_hive(ORIGIN_TABLE, from_dt=FROM_DT, to_dt=TO_DT)
    # print(dataset.take(1))
    transformers = parse_transform(trans_dic)
    # print(transformers)
    train_transformed, prd_transformed = transform_pipeline(dataset, transformers)
    print(train_transformed.first())

    write_to_hdfs(train_transformed, train_temp_path)
    write_to_hdfs(prd_transformed, prd_temp_path)

    hdfs_to_local(train_temp_path, 'spark_df_train')
    hdfs_to_local(prd_temp_path, 'spark_df_prd')
    gen_libsvm('spark_df_train', ORIGIN_TRAIN)
    gen_libsvm('spark_df_prd', ORIGIN_PRD)















