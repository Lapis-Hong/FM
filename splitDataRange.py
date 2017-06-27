def splitDataRange(filePath, trainDataPath, testDataPath):
    with open(trainDataPath, 'w') as f_train,\
          open(testDataPath, 'w') as f_test,\
           open(filePath) as trainData:
        line = trainData.readline()
        count = 1
        while line:
            if count % 5 == 0: # chose 20% as test data
                f_test.write(line)
            else:
                f_train.write(line)
            line = trainData.readline()
            count += 1
    return trainDataPath, testDataPath
