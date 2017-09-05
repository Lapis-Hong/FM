"""Using linux awk tool to do the data preprocess, much faster way. Recommend way"""
from __future__ import print_function
import subprocess
import os

from conf import *
from data_process import *
from data_process.util import *


def load_data_from_hdfs(hdfs_path, file_name):
    """move hdfs datasets to current dir"""
    try:
        subprocess.check_call("hadoop fs -text {0}/* > {1}".format(hdfs_path, file_name), shell=True)
        print('Already load original data {0} from hdfs path {1}!'.format(file_name, hdfs_path))
    except subprocess.CalledProcessError as e:
        print("Command Error:", end=' ')
        print(e)


def save_data_to_hdfs(file_name, hdfs_path):
    """move current dir datasets to hdfs"""
    subprocess.call("hadoop fs -mkdir -p {0}".format(hdfs_path), shell=True)
    subprocess.call("hadoop fs -put {0} {1}".format(file_name, hdfs_path), shell=True)
    print('Already move the {0} to hdfs path {1}'.format(file_name, hdfs_path))


@clock()
def remove_zero_sed(infile, outfile):
    """remove zero index:feature pairs like 3:0, using linux sed tool"""
    sed = "sed -e 's/\s*[0-9]*:0//g' "
    cmd = sed + infile + " > " + outfile
    print(cmd)
    subprocess.call(cmd, shell=True)


@clock()
def remove_zero(infile, outfile):
    """remove zero index:feature pairs like 3:0, using linux awk tool"""
    awk_command = 'awk \'{printf $1} {for(i=2; i<=NF; i++){if($i !~/:0/){printf " "$i}}} {print " "}\' '
    cmd = awk_command + infile + " > " + outfile
    print('The shell command is: {0}'.format(cmd))
    subprocess.check_call(cmd, shell=True)


@clock()
def relabel(infile, outfile):
    """change the label {1, 0} to {1, -1}, using linux awk tool"""
    awk_command = 'awk \'{if($1==0){$1=-1}}{print $0}\' '  # need a space
    cmd = awk_command + infile + " > " + outfile
    print('The shell command is: {0}'.format(cmd))
    subprocess.call(cmd, shell=True)


@clock('Successfully convert to the libfm format!')
def relabel_and_remove_zero(infile, outfile):
    """change the label and remove zero, using linux awk tool"""
    awk_command = 'awk \'{if($1==0){$1=-1}} {printf $1} {for(i=2; i<=NF; i++){if($i !~/:0/){printf " "$i}}} {print " "}\' '
    cmd = awk_command + infile + " > " + outfile
    print('The shell command is: {0}'.format(cmd))
    subprocess.check_call(cmd, shell=True)


@clock('Successfully generate the libsvm format!')
def gen_libsvm(infile, outfile):
    awk_command = 'awk \'{printf $1} {for(i=2; i<=NF; i++) {printf "  "i-1":"$i}} {print " "}\' '
    cmd = awk_command + infile + ">" + outfile
    print('The shell command is:{0}'.format(cmd))
    subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    make_path(DATA_DIR, MODEL_DIR)

    load_data_from_hdfs(FROM_HDFS_TRAIN, ORIGIN_TRAIN)
    load_data_from_hdfs(FROM_HDFS_PRD, ORIGIN_PRD)

    index_dic = get_new_index(ORIGIN_TRAIN)
    pickle.dump(index_dic, open(os.path.join(MODEL_DIR, 'index_dump'), 'wb'))

    # relabel(ORIGIN_TRAIN, 'temp')
    # remove_zero(ORIGIN_TRAIN, 'temp')
    # remove_zero_sed(ORIGIN_TRAIN, 'temp')
    relabel_and_remove_zero(ORIGIN_TRAIN, os.path.join(DATA_DIR, FM_TRAIN))
    split_data(FM_TRAIN, TRAIN, TEST, mode='overwrite')
