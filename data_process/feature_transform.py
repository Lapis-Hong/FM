"""Read original data from hive, do the feature transform directly using Spark
TODO: need to improve the performance and test for multiple days data
"""
import datetime
import time
import os
import tempfile
from collections import Counter

import pandas as pd
from pyspark import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.types import Row

from mysql import trans_dic
from transformer import *
from data_process import clock
from data_process.util import start_spark
from shell import load_data_from_hdfs

# test_dic = {'age': 'discretize(20)', 'sex': 'onehot', 'wait_pay': 'bucket(min,10,100,max);onehot()'}


def read_from_hive(table, dt):
    _, ss = start_spark()
    df = ss.sql('select * from {0} where dt={1}'.format(table, dt))
    print('Successfully read the data from hive, {0} rows, {1} cols'.format(df.count(), len(df.columns)))
    return df


def parse_transform(transform_dict):
    transform_list = []
    for field_name in transform_dict.keys():
        if transform_dict[field_name] is None:
            transform_list.append(('delete', field_name))
        else:
            transformer = transform_dict[field_name].split(';')
            for transform in transformer:
                if transform.startswith('discretize'):
                    start = transform.find('(') + 1
                    end = transform.find(')')
                    arg = transform[start:end]
                    transform_list.append(('discretizer', field_name, int(arg)))
                elif transform.startswith('bucket'):
                    start = transform.find('(') + 1
                    end = transform.find(')')
                    arg = transform[start:end]
                    arg = arg.replace('min', "-inf")
                    arg = arg.replace('max', "inf")
                    transform_list.append(('bucketizer', field_name, [float(w) for w in arg.split(',')]))
                elif transform.startswith('onehot'):
                    transform_list.append(('onehot', field_name))
                elif transform.startswith('standardnormalize'):
                    transform_list.append(('standard_scaler', field_name))
                elif transform.startswith('Untransform'):
                    pass
                else:
                    raise Exception('transform {} of field {} is not found, please check the transform'.format(transform, field_name))
    c = Counter(elem[0] for elem in transform_list)
    print(c)
    print('There are {} features need to transform'.format(len(transform_list)))
    return transform_list


@clock()
def transform_pipeline(data, transform):
    data_type = dict(data.dtypes)
    data_frame = data.fillna(0)  # alias for df.na.fill()
    for index, item in enumerate(transform):
        if data_type[item[1]] == 'string':
            data_frame = data_frame.withColumn(item[1], data_frame[item[1]].astype('double'))  # change the column type
        t0 = time.time()
        if item[0] == 'bucketizer':
            df = bucktizer(data_frame, item[1], 'temp', splits=item[2], drop=True)
        elif item[0] == 'discretizer':
            df = discretizer(data_frame, item[1], 'temp', numBuckets=item[2], drop=True)
        elif item[0] == 'onehot':
            df = onehot(data_frame, item[1], 'temp', drop=True)
        elif item[0] == 'standard_scaler':
            df_temp = vector_assembler(data_frame, [item[1]], 'Vector', drop=True)  # convert double to the Vectors type
            df = standard_scaler(df_temp, 'Vector', 'temp', drop=True)
        elif item[0] == 'delete':
            df = data_frame.drop(item[1])
        else:
            raise TypeError('Unknown transfrom type, please check the input')
        data_frame = df.withColumnRenamed('temp', item[1])
        print('The {}th feature {}; transform {} has finished, take {} sec.'.format(index+1, item[1], item[0], time.time()-t0))
    print('Feature transform has done!')
    print(data_frame.columns)
    return data_frame


@clock()
def write_to_hdfs(data, path=os.path.join(tempfile.mkdtemp(), 'data')):
    data.write.mode('error').parquet(path)
    print('Already save data to {}'.format(path))
    return path


if __name__ == '__main__':
    path = 'pre_credit_user_fm'
    begin = datetime.date(2017, 4, 21)
    end = datetime.date(2017, 7, 20)
    for i in range((end-begin).days+1):
        day = begin + datetime.timedelta(days=i)
        dt = day.strftime("%Y%m%d")
        dataset = read_from_hive('temp_jrd.pre_credit_user_feature', dt)
        # print(dataset.take(1))
        transformers = parse_transform(trans_dic)
        # print(transformers)
        data_transformed = transform_pipeline(dataset, transformers)
        print(data_transformed.first())
        # hdfs_path = write_to_hdfs(data_transformed, 'pre_credit_user_fm')
        data_transformed.write.mode('append').parquet(path)
        print('{0} feature transform has finished'.format(dt))

    load_data_from_hdfs(path, 'spark df')













