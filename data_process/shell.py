from __future__ import print_function
import subprocess
from conf import *
from data_process import clock


def load_data_from_hdfs(hdfs_path, file_name):
    """move the hdfs datasets to the current dir"""
    try:
        subprocess.check_call("hadoop fs -text {0} > {1}".format(hdfs_path, file_name), shell=True)
        print('Already load data from hdfs!')
    except subprocess.CalledProcessError as e:
        print("Command Error:", end=' ')
        print(e)


def save_data_to_hdfs(hdfs_path, file_name):
    """move the current dir datasets to the hdfs"""
    subprocess.call("hadoop fs -mkdir {0}".format(hdfs_path), shell=True)
    subprocess.call("hadoop fs -put {0} {1}".format(file_name, hdfs_path), shell=True)
    print('Already move the {0} to hdfs path {1}'.format(file_name, hdfs_path))


@clock()
def remove_zero_sed(infile, outfile):
    sed = "sed -e 's/\s*[0-9]*:0//g' "
    cmd = sed + infile + ">" + outfile
    print(cmd)
    subprocess.call(cmd, shell=True)


@clock()
def remove_zero(infile, outfile):
    awk_command = 'awk \'{printf $1} {for(i=2; i<=NF; i++){if($i ~/:1/){printf " "$i}}} {print " "}\' '
    cmd = awk_command + infile + ">" + outfile
    subprocess.check_call(cmd, shell=True)


@clock()
def relabel(infile, outfile):
    awk_command = 'awk \'{if($1==0){$1=-1}}{print $0}\' '  # need a space
    cmd = awk_command + infile + ">" + outfile
    subprocess.call(cmd, shell=True)


@clock()
def relabel_and_remove_zero(infile, outfile):
    awk_command = 'awk \'{if($1==0){$1=-1}} {printf $1} {for(i=2; i<=NF; i++){if($i ~/:1/){printf " "$i}}} {print " "}\' '
    cmd = awk_command + infile + ">" + outfile
    subprocess.check_call(cmd, shell=True)


def feature_target_split():
    pass


def gen_libsvm(infile, outfile):
    awk_command = 'awk \'{printf $1} {for(i=2; i<=NF; i++) {printf "  "i-1":"$i}} {print " "}\' '
    cmd = awk_command + infile + ">" + outfile
    subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    load_data_from_hdfs(HDFS_PATH, ORIGIN_TRAIN)
    # relabel(LIBSVM_PRD_FILE, 'remove')
    # remove_zero(LIBSVM_PRD_FILE, 'remove1')
    # remove_zero_sed(LIBSVM_PRD_FILE, 'remove3')
    relabel_and_remove_zero(ORIGIN_TRAIN, FM_TRAIN)



