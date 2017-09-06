from conf import *
from data_process import shell, python, spark
import train
import embedding


"""STEP 1: Data Process"""
if DATA_PROCESS == 'shell':
    shell.main()
elif DATA_PROCESS == 'spark':
    spark.main()
else:
    python.main()

"""STEP 2: Train Model"""
train.main()

"""STEP 3: Embedding Process"""
embedding.main()








