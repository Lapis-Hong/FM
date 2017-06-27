****************************************************************************************
安装libFM
1.程序包在集群master上，路径：/home/apps/jarfile/lxt/libfm
2.make all
3.设置环境变量：export LIBFM_PATH=/home/apps/jarfile/lxt/libfm/bin/

python wrapper for libFM
路径：pywFM/__init__.py

****************************************************************************************
factorization machine embedding:
注：以testfeatureengine.featuremetadata where modelname = 'wphorder_used_fm'为例；
	每一个fieldname表示一个field。对该field进行onehot，获得多个feature
	indexnum特指featuremetadata中的indexnum
	隐向量维数用k_dim表示

一.数据准备
1.特征工程进行transform，生成经过onehot的train数据与prd数据

2.记录每一个field经过onehot生成feature的indexnum
	例：indexnum=89的field是‘vmark_total’
		经过onehot之后生成feature对应的indexnum范围是499~518；
		indexnum=90的field是‘vmark_next_upgrade’
		经过onehot之后生成feature对应的indexnum范围是519~535；
		indexnum=90的field是‘point_available’
		经过onehot之后生成feature对应的indexnum范围是536~555；
		
3.特征工程生成的libsvm数据，feature的序号重新排序
	特诊工程生成的libsvm数据，feature的序号是用表featuremetadata的indexnum。因此做onehot之后feature的序号不连续。
	所以对libsvm数据中的feature序号从0进行重新排序，相应的要记录生成feature对应的新范围。
	例：
		libsvm生成的train数据：1.0 499:0 500:0 501:0 502:0 503:0 504:0 505:1 506:0 507:0 508:0 509:0 510:0 511:0 512:0 513:0 514:0 515:0 516:0 517:0 518:0 519:0 520:0 521:0 522:0 523:0 524:0 525:0 526:0 527:1 528:0 529:0 530:0 531:0 532:0 533:0 534:0 535:0 536:0 537:0 538:0 539:0 540:0 541:0 542:1 543:0 544:0 545:0 546:0 547:0 548:0 549:0 550:0 551:0 552:0 553:0 554:0 555:0 
		重新排序生成的数据：   1.0 0:0   1:0   2:0   3:0   4:0   5:0   6:1   7:0   8:0   9:0   10:0  11:0  12:0  13:0  14:0  15:0  16:0  17:0  18:0  19:0  20:0  21:0  22:0  23:0  24:0  25:0  26:0  27:0  28:1  29:0  30:0  31:0  32:0  33:0  34:0  35:0  36:0  37:0  38:0  39:0  40:0  41:0  42:0  43:1  44:0  45:0  46:0  47:0  48:0  49:0  50:0  51:0  52:0  53:0  54:0  55:0  56:0  
		indexnum=89的field是‘vmark_total’
		经过onehot之后生成feature对应的新序号范围是0~19；
		indexnum=90的field是‘vmark_next_upgrade’
		经过onehot之后生成feature对应的新序号范围是20~36；
		indexnum=90的field是‘point_available’
		经过onehot之后生成feature对应的新序号范围是37~56；
		
4.libsvm数据中去除value为0的特征
	例：
		libFM使用的train数据：1.0 6:1 28:1 43:1

二.训练

使用libFM进行训练，获得每一个特征对应的隐向量
隐向量存入文件中

三.embedding

1.保留prd数据的‘order_id’与需要做embedding的feature序号
	例：
		1234567890 6:1 28:1 43:1
		1234567891 6:1 43:1

2.prd数据中的feature与feature范围匹配。如果匹配，用feature的序号查找隐向量；如果没有匹配，用k_dim个0补充
	例：三个field，经过onehot，生成feature对应的序号范围是(0,19),(20,36).(37,56)
		1234567890 (隐向量) (隐向量)   (隐向量) 
		1234567891 (隐向量) (k_dim个0) (隐向量) 

3.得到新特征存入hive中。order_id为主键，与其他数据进行join

****************************************************************************************
改进：
1.有些field的value只有0或1，是否需要作为数据的一部分，进行FM的训练。
2.有些field进行离散化后，只能离散出少数的feature(如一个field离散成5个feature)。如果隐向量维数大于feature的数量，是否有必要进行embedding。
3.field进行onehot之后，生成feature对应的indexnum范围需人为记录。
4.唯品花已开已用数据不全