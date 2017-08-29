****************************************************************************************
所需环境
1. C++library:
安装libFM，libFFM, 并设置环境变量LIBFM_PATH, LIBFFM_PATH
2. Python 环境:
python 2.6及以上, embedding过程需要python 2.6, 已搭建虚拟环境 py2.6, source activate py2.6 进入, source deactivate 退出
所需python库: numpy，scipy，pandas，sklearn，pyspark, fastFM（安装失败）

参考
libFM: http://www.libfm.org
fastFM: http://ibayer.github.io/fastFM/#
libffm: http://www.csie.ntu.edu.tw/~cjlin/libffm

****************************************************************************************
FM embedding

流程
HIVE > HDFS > 本地目录训练 > 写入HIVE or HDFS

一.数据准备
1.特征变换配置:
  观察所有特征的分布情况
  目前对适合离散化的连续变量配bucket和discretize的离散化处理
  绝大部分为0的连续变量离散化后的特征数极少, 因此直接配standardnormalize
  多类别变量配onehot
  flag型变量配untransform
  tips: FM 适合类别数较多的category feature, onehot后足够稀疏, embedding效果可能更优

2.原始数据:
  特征工程进行transform后生成libsvm格式的train数据与prd数据。
  注意每一行的feature数量要相同，在featuremetadata配置特征变换时将defaultvalue设置为0即可
  经过onehot变换的数据会embedding, 生成的索引会在原始feature index之后,
  若不希望embedding的feature配其他特征变换即可。

3.数据预处理: 只针对train数据
  target由{1, 0}转化为{1, -1}
  去除index:value中value值为0的数据, 减少数据量, 加快训练速度
  获得索引映射关系字典 原始libsvm中索引映射成从0开始的索引, 原始index:新index, 如{'56':'0', '57':'1'...}, 为了之后embedding的需要
  划分libfm的训练测试集

  提供了三种方式: shell, python, spark 性能spark >> shell > python  此处spark仍为单机多线程模式
  性能测试: 8G 原始数据预处理耗时分别为 spark(130s) shell(700s) python(780s)
		

二.训练

提供了三种接口: libfm, fastFM以及libffm(libfm已成熟, fastFM安装有问题, libffm数据预处理部分还没完成)
训练时间主要与迭代次数, 学习率和隐向量维数有关, 可通过降低迭代次数, 提高学习率以及降低隐向量维数来加快训练


三.embedding

1.逻辑:
假如类别1有5类, onehot后生成的一个样本是[0, 1, 0, 0, 0], 那么对应激活的隐向量为1所对应的index值的隐向量
因此，对于每一个样本而言，每一个类别有且只有一个被激活，embedding的逻辑就是逢1便填入隐向量latent[index]

2.提供了两种embedding方式，通过在embedding.py里函数中isappend参数设置
  isappend=True 在原有feature基础上增加embedding产生的新feature
  isappend=False 只写入embedding产生的新feature

3.隐向量写入hive,
(1)embedding数据写入hive与其他数据join供其他模型使用; 需要order_id和target字段都写入hive, 方便后续使用数据
此时格式需dataframe格式
(2)embedding数据直接写入HDFS, 此时需要写入两份数据, 一份train, 一份prd
此时格式需libsvm格式

****************************************************************************************
四.使用指南

1.一键使用:
  切到myFM／目录, 修改conf.py里面的配置参数 然后 $python main.py

2.各模块独立使用:
  数据预处理 $python data_process/spark.py (或者shell.py or python.py)
  训练 $python train.py 可命令行调参 具体usage 参见 python train.py -h
  embeeding $python embedding.py


****************************************************************************************
TODO:
libffm的数据预处理: 需要field:feature:value格式
