import time
time_start = time.time()
from removeZeroValue import *
data_libsvm_path = 'fmtrain20170403.txt' # original data
data_temp_path = 'fmtrain20170403.temp2' # modified index data
data_libfm_path = 'fmtrain20170403.libfm' # data format for fm

data_libsvm = open(data_libsvm_path)
data_temp = open(data_temp_path, 'w')
data_libfm = open(data_libfm_path, 'w')

data_temp_list = []
data_libfm_list = []
count = 0
for line in data_libsvm.readlines():
    line = line.strip('\n').split(' ') # convert string to list
    label = line.pop(0)
    data_temp_list.append(str(label))
    data_libfm_list.append(str(label))
    feature_index_new = 1
    for element in line:
        temp = element.split(':') # string format
        feature_value = float(temp[-1])
        data_temp_list.append(' ' + str(feature_index_new) + ':' + str(feature_value))
        if feature_value != 0:
            data_libfm_list.append(' ' + str(feature_index_new) + ':' + str(feature_value) )
        feature_index_new += 1
    data_temp_list.append('\n')
    data_libfm_list.append('\n')
    count += 1
    if count%10000 == 0:
        print(count)

data_temp.write(''.join(data_temp_list))
data_libfm.write(''.join(data_libfm_list))
data_libsvm.close()
data_temp.close()
data_libfm.close()
time_end = time.time()
print(time_end - time_start)
