****************************************************************************************
所需环境
1. C++library:
安装libFM，libFFM, 并设置环境变量LIBFM_PATH, LIBFFM_PATH
2. Python 环境:
python 2.6及以上, embedding过程需要python 2.6, 已搭建虚拟环境 py2.6, source activate py2.6 进入, source deactivate 退出
所需python库: numpy, scipy, pandas, argparse, sklearn, pyspark, fastFM（安装失败）

参考资料
libFM: http://www.libfm.org
fastFM: http://ibayer.github.io/fastFM/#
libffm: http://www.csie.ntu.edu.tw/~cjlin/libffm

****************************************************************************************
FM embedding

流程
HIVE > HDFS > 本地训练 > 写入HIVE & HDFS

一.数据准备
1.特征变换配置:
  观察所有特征的分布情况选择最佳变换
  目前对适合离散化的连续变量配bucket和discretize的离散化处理
  绝大部分为0的连续变量离散化后的特征数极少, 因此直接配standardnormalize
  多类别变量配onehot
  flag型变量配untransform
  tips: FM 适合类别数较多的category feature, onehot后足够稀疏, embedding效果可能更优

2.原始数据:
  特征工程进行transform后生成libsvm格式的train数据与prd数据
  注意每一行的feature数量要相同，在featuremetadata配置特征变换时将defaultvalue设置为0即可
  经过onehot变换的数据才会embedding, 生成的索引会在原始feature index(564)之后,
  若不希望embedding的feature配其他特征变换即可。
  (feature_transform.py中提供了绕过特征工程直接调用spark做transform的方法,
  生成dataframe格式数据存入hdfs然后再复制到本地生成libsvm格式数据。目前少量特征变换已跑通, 但是大量的特征变化慢且超内存)
  load data from hdfs to local
  性能测试:
  11G 100s

3.数据预处理: 只针对train数据
  target由{1, 0}转化为{1, -1}
  去除index:value中value值为0的数据, 减少数据量, 加快训练速度
  获得索引映射关系字典 原始libsvm中索引映射成从0开始的索引, 原始index:新index, 如{56:0, 57:1 ...},
  为了之后embedding的需要划分libfm的训练集测试集 性能(11G 90s)

  提供了三种方式: shell, python, spark 性能 python > spark >> shell
  此处python 用了并发, spark仍为单机多线程模式, shell单线程 故默认使用python并发模式
  性能测试:
  原始数据预处理耗时:
  8G  spark(130s) shell(700s) python(780s)
  11G python(160s并发) spark(210s) shell(1100s) python(1300s单线程)

二.训练

提供了三种接口: libfm, fastFM以及libffm(libfm已成熟, fastFM安装有问题, libffm数据预处理部分还没完成)
训练时间主要与迭代次数, 学习率和隐向量维数有关, 可通过降低迭代次数, 提高学习率以及降低隐向量维数来加快训练
因为最终目的是embedding而不是直接预测, 每次迭代准确率差异极小, 故可减少迭代次数, 实现scalable目标

性能测试:
libfm默认参数下 11G mcmc(3000s) als(3000s)

三.embedding

1.逻辑:
简单考虑两个类别变量两条数据的情况, 每个类别变量4个取值, onehot后生成10个feature如下:
   f1 f2 f3 f4  f5 f6 f7 f8
1) 0  1  0  0    1  0  0  0
2) 1  0  0  0    0  0  0  1
取隐向量维度k=2, 每个特征学习到一个隐向量, 则生成的隐向量矩阵 10*2 如下:
f1 <->  v11 v12
f2 <->  v21 v22
f3 <->  v31 v32
f4 <->  v41 v42
f5 <->  v51 v52
f6 <->  v61 v62
f7 <->  v71 v72
f8 <->  v81 v82
embedding后生成的数据如下: []表示可选, 即是否append
1) 0  1  0  0  1  0  0  0  --> [0  1  0  0  1  0  0  0] v21 v22  v51 v52
2) 1  0  0  0  0  0  0  1  --> [1  0  0  0  0  0  0  1] v11 v12  v81 v82

上述类别1有4类, onehot后生成的一个样本是[0, 1, 0, 0], 那么对应激活的隐向量为1所对应的index:2的隐向量(v21, v22)
因此，对于每一个样本而言，每一个类别有且只有一个被激活，embedding的逻辑就是逢1便填入隐向量latent[index]

2.embedding维度选择
tips1: 计算能力足够的情况下, 尽可能选较大的k值
tips2: 变量类别普遍多的情况下, k值相应要大一些

3.提供了两种embedding方式，通过在embedding.py里函数中isappend参数设置
  isappend=True 在原有feature基础上增加embedding产生的新feature
  isappend=False 只写入embedding产生的新feature

4.embedding数据写入HIVE & HDFS
基本思路: 分批生成rdd, 再转成spark df写入hive & 分批生成rdd写入hdfs 或写入本地再传到hdfs
(1)embedding数据写入hive与其他数据join供其他模型使用; 需要order_id和target字段都写入hive, 方便后续使用数据
此时格式需dataframe格式
(2)embedding数据直接写入HDFS, 此时需要写入两份数据, 一份train, 一份prd
此时格式需libsvm格式

性能测试: (此部分是整个project的性能瓶颈, spark写入hive非常慢, 此处不知r-server上是否使用集群)
写入HIVE: 11G 262万行 33360s
写入HDFS: 11G 262万行 两份数据train和prd 1666s(python 多进程先写到local再转到hdfs)  20000s(直接用spark写入hdfs)

****************************************************************************************
四.使用指南

切换python2.6环境: $source activate py2.6

1.一键使用:
  切到目录 FM／, 修改conf.py里面的配置参数 然后 $python main.py

2.各模块独立使用:
  修改配置文件 $vim conf.py
  数据预处理  $python data_process/spark.py (或者shell.py or python.py)
  训练  $python train.py (可命令行调参 具体usage 参见 python train.py -h)
  embedding  $python embedding.py

3.目录内容:
  Data／ 生成的libfm数据及其训练测试集
  temp／ 原始数据split产生的分块数据, 包含train-part* 和 prd-part*  为了python并发读写加快速度
  Model／ 模型参数, index mapping 和 latent dump
  Embedding／ embedding data 分块数据集, 可能包括trian-part* prd-part* 和train_prd-part*
              当本地空间不足时, 可以通过设置keep_local参数来决定是否保留本地数据

****************************************************************************************
TODO:
libffm的数据预处理: 需要field:feature:value格式
feature_transform.py 大数据下内存报错和性能问题

