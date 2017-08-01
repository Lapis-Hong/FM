
import time
import pandas as pd
from functools import wraps
from pyspark import *
from pyspark.ml.feature import *
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.types import Row

from mysql import config
from transformer import *
from spark import start_spark
from data_process import clock

# test_dic = {'age': 'discretize(20)', 'sex': 'onehot', 'wait_pay': 'bucket(min,10,100,max);onehot()'}
def read_from_hive():
    _, ss = start_spark()
    dataset = ss.sql('select * from temp_jrd.pre_credit_user_feature where dt=20170707')
    return dataset



def parse_transform(transform_dict):
    transform_list = []
    for field_name in transform_dict.keys():
        transformer = transform_dict[field_name].split(';')
        for transform in transformer:
            if transform.startswith('discretize'):
                start = transform.find('(') + 1
                end = transform.find(')')
                arg = transform[start:end]
                transform_list.append(('discretizer', field_name, arg))
            elif transform.startswith('bucket'):
                start = transform.find('(') + 1
                end = transform.find(')')
                arg = transform[start:end]
                arg = arg.replace('min', '-float("inf")')
                arg = arg.replace('max', 'float("inf")')
                transform_list.append(('bucketizer', field_name, arg.split(',')))
            elif transform.startswith('onehot'):
                transform_list.append(('onehot', field_name))
            elif transform.startswith('standardnormalize'):
                transform_list.append(('standard_scaler', field_name))
            elif transform.startswith('Untransform'):
                pass
            elif transform is None:
                transform_list.append(('delete', field_name))
            else:
                raise Exception('transform {} of field {} is not found, please check the transform'.format(transform, field_name))
    print('There are {} features need to transform'.format(len(transform_list)))
    return transform_list


@clock()
def transform_pipeline(data, transform):
    for index, item in enumerate(transform):
        t0 = time.time()
        if item[0] == 'bucktizer':
            df = bucktizer(data, inputCol=item[1], outputCol=item[1]+'trans', splits=item[2], drop=True)
        elif item[0] == 'discretizer':
            df = discretizer(data, inputCol=item[1], outputCol=item[1]+'trans', numBuckets=item[2], drop=True)
        elif item[0] == 'onehot':
            df = onehot(data, inputCol=item[1], outputCol=item[1]+'trans', drop=True)
        elif item[0] == 'standard_scaler':
            df = standard_scaler(data, inputCol=item[1], outputCol=item[1]+'trans', drop=True)
        elif item[0] == 'delete':
            df = df.drop(item[1])
        data = df
        print('The {}th feature: {} transform: {} has finished, take {} sec.'.format(index, item[0], item[1], time.time()-t0))
    print('Feature transform has done!')
    return data

def write_data():
    pass


if __name__ == '__main__':
    # conf = SparkConf().setAppName("testSpark").setMaster('yarn').set("spark.executor.memory", "10g")
    # sc = SparkContext(conf=conf)
    # spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    dataset = read_from_hive()
    start_spark()
    transformers = parse_transform(trans_dic)
    data_transformed = transform_pipeline(dataset, transformers)
    print(data_transformed.first())












