"""
Wrap the Spark feature transformers to use for our feature transformation
"""
import pandas as pd
from functools import wraps
from pyspark import *
from pyspark.ml.feature import *
from data_process.util import start_spark

# vectorize = udf(lambda vs: Vectors.dense(vs), VectorUDT())
# assembler = VectorAssembler(
#     inputCols=["features[{0}]".format(i) for i in range(n)],
#     outputCol="features_vector")


def pandas_to_spark(file_name):
    _, ss = start_spark()
    df = pd.read_csv(file_name)  # null is NaN in pandas
    if df.isnull().any().sum() > 0:
        category_cols = df.columns[df.dtypes == object]  # object is mix type
        if len(category_cols) > 0:
            df.fillna('0', inplace=True)
            numerical_cols = df.columns[df.dtypes != object]
            df = df[numerical_cols].astype(float)  # change the string '0' to float 0.0
            spark_df = ss.createDataFrame(df)
        else:
            spark_df = ss.createDataFrame(df)
    else:
        spark_df = ss.createDataFrame(df)
    return spark_df
# df = spark.read.format("csv").options(header="true").load("fmtrain20170704")
# df.dropna(axis=1, how='all')
# df[df==0] = np.nan  # can take 0 to NaN
# pandas_df.info()


def multitransform(func):  # need modify
    @wraps(func)
    def multitransformer(dataframe, input_cols, output_cols, **kwargs):  # inputCols can not be keyword args
        if isinstance(input_cols, str):
            df = func(dataframe, inputCol=input_cols, outputCol=output_cols, **kwargs)
            return df
        elif len(input_cols) != len(output_cols):
            raise TypeError('inputCols and outputCols should be the same dimension')
            for inputCol, outputCol in zip(inputCols, outputCols):
                df = func(dataframe, inputCol=inputCol, outputCol=outputCol, **kwargs)
                dataframe = df
            return df
    return multitransformer


@multitransform
def binarizer(dataFrame, threshold=0.0, inputCol=None, outputCol=None, drop=False):
    binarizer = Binarizer(threshold=threshold, inputCol=inputCol, outputCol=outputCol)
    df = binarizer.transform(dataFrame)
    if drop:
        df = df.drop(inputCol)
    return df


@multitransform
def string_indexer(dataFrame, inputCol, outputCol, handleInvalid='error'):
    indexer = StringIndexer(inputCol=inputCol, outputCol=outputCol, handleInvalid=handleInvalid)
    df = indexer.fit(dataFrame).transform(dataFrame)
    return df


@multitransform
def index_to_string(dataFrame, inputCol, outputCol, labels):
    converter = IndexToString(inputCol=inputCol, outputCol=outputCol, labels=labels)
    df = converter.transform(dataFrame)
    return df


@multitransform
def onehot(dataFrame, inputCol, outputCol, dropLast=False, drop=False):
    stringIndexer = StringIndexer(inputCol=inputCol, outputCol='indexed')
    indexed = stringIndexer.fit(dataFrame).transform(dataFrame)
    encoder = OneHotEncoder(dropLast=dropLast, inputCol='indexed', outputCol=outputCol)
    df = encoder.transform(indexed)
    df = df.drop('indexed')
    if drop:
        df = df.drop(inputCol)
    return df


@multitransform
def normalizer(dataFrame, inputCol, outputCol, p=2.0):
    normalizer = Normalizer(inputCol=inputCol, outputCol=outputCol, p=p)
    df = normalizer.transform(dataFrame)
    return df


@multitransform
def standard_scaler(dataFrame, inputCol, outputCol, withStd=True, withMean=True, drop=False):
    scaler = StandardScaler(inputCol=inputCol, outputCol=outputCol,
                            withStd=withStd, withMean=withMean)
    df = scaler.fit(dataFrame).transform(dataFrame)
    if drop:
        df = df.drop(inputCol)
    return df


@multitransform
def min_max_scaler(dataFrame, inputCol, outputCol, min=0.0, max=1.0):
    scaler = MinMaxScaler(inputCol=inputCol, outputCol=outputCol, min=min, max=max)
    df = scaler.fit(dataFrame).transform(dataFrame)
    return df


@multitransform
def max_abs_scaler(dataFrame, inputCol, outputCol, ):
    """does not destroy any sparsity"""
    scaler = MaxAbsScaler(inputCol=inputCol, outputCol=outputCol)
    df = scaler.fit(dataFrame).transform(dataFrame)
    return df


@multitransform
def bucktizer(dataFrame, inputCol, outputCol, splits, handleInvalid='error', drop=False):
    # splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]
    bucketizer = Bucketizer(splits=splits, inputCol=inputCol, outputCol=outputCol, handleInvalid=handleInvalid)
    df = bucketizer.transform(dataFrame)
    if drop:
        df = df.drop(inputCol)
    return df


def vector_assembler(dataFrame, inputCols, outputCol, drop=False):
    assembler = VectorAssembler(inputCols=inputCols, outputCol=outputCol)
    df = assembler.transform(dataFrame)
    if drop:
        if isinstance(inputCols, basestring):
            df = df.drop(inputCols)
        else:
            for col in inputCols:
                df = df.drop(col)
    return df


@multitransform
def discretizer(dataFrame, inputCol, outputCol, numBuckets=2, relativeError=0.001, handleInvalid='error', drop=False):
    discretizer = QuantileDiscretizer(numBuckets=numBuckets, inputCol=inputCol, outputCol=outputCol,
                                      relativeError=relativeError, handleInvalid=handleInvalid)
    df = discretizer.fit(dataFrame).transform(dataFrame)
    if drop:
        df = df.drop(inputCol)
    return df


@multitransform
def untransform(dataFrame, inputCol, outputCol):
    return dataFrame.withColumn(outputCol, dataFrame[inputCol])


def vector_indexer(dataFrame, inputCol, outputCol, maxCategories=20):
    """
    - Automatically identify category features
    - Index all features, if all features are categorical
    """
    indexer = VectorIndexer(inputCol=inputCol, outputCol=outputCol, maxCategories=maxCategories)
    df = indexer.fit(dataFrame).transform(dataFrame)
    return df


# feature selection functions
def vector_slicer(dataFrame, inputCol, outputCol, indices):
    slicer = VectorSlicer(inputCol=inputCol, outputCol=outputCol, indices=indices)
    df = slicer.transform(dataFrame)
    return df


def r_formula(dataFrame, formula, featuresCol, labelCol):
    # #dataset = spark.createDataFrame(
    #     [(7, "US", 18, 1.0),
    #      (8, "CA", 12, 0.0),
    #      (9, "NZ", 15, 0.0)],
    #     ["id", "country", "hour", "clicked"])
    formula = RFormula(formula=formula, featuresCol=featuresCol, labelCol=labelCol)
    df = formula.fit(dataFrame).transform(dataFrame)
    return df


def chisq_selector(dataFrame, numTopFeatures, featuresCol, outputCol, labelCol):
    '''cor with label, chose the best predictable features'''
    # df = spark.createDataFrame([
    #     (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    #     (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    #     (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])
    selector = ChiSqSelector(numTopFeatures=numTopFeatures, featuresCol=featuresCol,
                             outputCol=outputCol, labelCol=labelCol)
    df = selector.fit(dataFrame).transform(dataFrame)
    return df

# pipeline way to do
# cols = spark_df.columns
# t1 = QuantileDiscretizer(numBuckets=20, inputCol=cols[4], outputCol='f4_temp')
# t2 = OneHotEncoder(inputCol='f4_temp', outputCol='f4')
# #t3 = OneHotEncoder(inputCol=cols[5], outputCol='f5')
# t4 = OneHotEncoder(inputCol=cols[6], outputCol='f6')
# #t3 = QuantileDiscretizer(numBuckets=20, inputCol='used_mark', outputCol='f3')
# #t4 = OneHotEncoder(inputCol='flag_cmnc', outputCol='f4')
# pipeline = Pipeline(stages=[t1,t2,t4])
#
# estimator = pipeline.fit(spark_df)
# transformer = estimator.transform(spark_df)
# transformer.show()


