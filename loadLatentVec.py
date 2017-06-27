# libFM save latent Vector in .txt; and save latent Vector in List
def latentVecLoad(filePath):
    with open(filePath) as model:
        f = model.read()
        # latentVec = [[str(w) for w in line] for line in latentVec]
    return eval(f)
'''
[[-0.145157, -0.101264, 0.0266148, -0.123098, -0.10167, 0.10267, 0.0328743, -0.20144], [-0.105264, -0.137794, 
'''
'''
>>> print(latentVec[0])
[['0.0493568', '0.0268083', '-0.0839762', '0.0366637', '0.0398121', '-0.205237', '0.203065', '-0.0830106']]
'''


def latentVecEmbedding(filePath, latentVec, indexList):
    new_data = []
    with open(filePath) as trainData:
        for line in trainData:
            embedding_feature = []
            temp = line.strip('\n').split(' ')
            embedding_feature.append(temp.pop(0)) # add target to the first col
            feature_index = [int(w.split(':')[0]) for w in temp if int(w.split(':')[0]) in indexList] # save the embedding feature from training data(with value 1)
            for embedding_ind in indexList:
                if embedding_ind in feature_index:
                    embedding_feature.extend(latentVec[embedding_ind]) # index from zero, and float format data for latentVec
                else:
                    embedding_feature.extend([0.0]*len(latentVec[0]))
            new_data.append(embedding_feature)
    return new_data

# create the embedding data
# def latentVecEmbedding(target, feature_index, latentVec, embeddingIndex):
#     new_data = []
#     sample_id = 0
#     target_index = 0
#     for sample in feature_index:
#         new_feature = []
#         temp_target = target.pop(0)
#         new_feature.append(temp_target)
#         for embeddingRange in embeddingIndex:
#             flag = False
#             for index in sample:
#                 if int(index) in embeddingRange:
#                     target_index = int(index)
#                     flag = True
#             if (flag == True):
#                 temp = latentVec[target_index]
#                 for i in temp[0]:
#                     new_feature.append(float(i))
#             else:
#                 for i in range(8):
#                     new_feature.append(0.0)
#         new_data.append(new_feature)
#         sample_id += 1
#         target_index += 1
#     return new_data


def tempLoadData(filePath):
    data = open(filePath)
    line = data.readline()
    DataList = []
    sample_id = 0
    while line:
        temp = line.strip('\n').split(',')
        tempDataList = []
        tempDataList.append(sample_id)
        for element in temp:
            tempDataList.append(float(element))
        DataList.append(tempDataList)
        line = data.readline()
        sample_id += 1
    data.close()
    print(DataList[0])
    return DataList


def dataSaveTxt(data, filePath):
    embeddingData = open(filePath, 'w')
    dataCount = -1
    for sample in data:
        dataCount += 1
        sampleCount = -1
        for element in sample:
            sampleCount += 1
            embeddingData.write(str(element))
            if (sampleCount != (len(sample) - 1)):
                embeddingData.write(',')
        if (dataCount != (len(data) - 1)):
            embeddingData.write('\n')
    embeddingData.close()
    return 1


def printRows(rows):
    if rows:
        for row in rows:
            print(row)


def genCmdLine(trainDataPath, testDataPath):
    cmdline = []
    # essential
    cmdline.extend(["./libFM", "-task", "c", "-train", trainDataPath,
                    "-test", testDataPath, "-dim", "1,1,8"])
    # parameter
    cmdline.extend(["-iter","234", "-method", "sgd", "-learn_rate",
                    "0.01", "-regular", "'0,0,0.01'", "-init_stdev", "0.1"])
    # save model and predict
    cmdline.extend(["-save_model","model","-out", "output"])
    return cmdline
