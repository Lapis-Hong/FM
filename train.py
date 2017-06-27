
import pyFM
from loadLatentVec import *
from splitDataRange import *
originData = []
originData.append('/Users/lapis-hong/Desktop/fmtrain20170403.libfm')
#originData.append('/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170404.libsvm')
#originData.append('/home/apps/jarfile/lxt/libfm/trainAndpredict/prepareData/fmtrain20170405.libsvm')
trainData, testData = splitDataRange(originData[0], 'ratings.train', 'ratings.test')
# set the parameter of FM
fm = pyFM.FM(
    task='classification',
    num_iter=100,
    k2=8,
    learning_method='sgd',
    learn_rate=0.01,
    r2_regularization=0.01,
    temp_path='/Users/lapis-hong/Desktop'
)
# set the path of trainData; testData; latent Vector
model = fm.run(
    trainset = trainData,
    testset = testData)
    #model_fd_name = 'latent')
print("FM training is finished.")


# save the latent vectors
with open('/Users/lapis-hong/Desktop/latent.txt', 'w') as latent:
    latent.write(str(model.pairwise_interactions))
