"""A Python wrapper for the libffm library.
References
----------
- `libffm: open source C++ library`
- `Github <https://github.com/guestwalk/libffm>`
<http://www.csie.ntu.edu.tw/~r01922136/libffm/`
- `FFM formulation details <http://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`
- `Criteo winning submission details <http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf>`
"""
import os
import subprocess


class Libffm:
    """
    `ffm-train'
    usage: ffm-train [options] training_set_file [model_file]
    options:
        -l <lambda>: set regularization parameter (default 0.00002)
        -k <factor>: set number of latent factors (default 4)
        -t <iteration>: set number of iterations (default 15)
        -r <eta>: set learning rate (default 0.2)
        -s <nr_threads>: set number of threads (default 1)
        -p <path>: set path to the validation set
        --quiet: quiet model (no output)
        --no-norm: disable instance-wise normalization
        --auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)
    
        By default we do instance-wise normalization. That is, we normalize the 2-norm of each instance to 1. You can use
        `--no-norm' to disable this function.
        
        A binary file `training_set_file.bin' will be generated to store the data in binary format.
    
        Because FFM usually need early stopping for better test performance, we provide an option `--auto-stop' to stop at
        the iteration that achieves the best validation loss. Note that you need to provide a validation set with `-p' when
        you use this option.
        
    `ffm-predict'
    usage: ffm-predict test_file model_file output_file  
    """
    def __init__(self,
                 dim=4,
                 iter_num=15,
                 learn_rate=0.2,
                 regularization=0.00002,
                 validation_set=None,
                 threads=1,
                 silent=False,
                 no_norm=False,
                 auto_stop=False):
        self.__dim = dim
        self.__iter_num = iter_num
        self.__learn_rate = learn_rate
        self.__regularization = regularization
        self.__validation_set = validation_set
        self.__threads = threads
        self.__silent = silent
        self.__no_norm = no_norm
        self.__auto_stop = auto_stop
        self.__libffm_path = os.environ.get('LIBFFM_PATH')  # gets libffm path
        if self.__libffm_path is None:
            raise OSError("`LIBFFM_PATH` is not set. Please install libffm and set the path variable "
                          "(https://github.com/guestwalk/libffm).")
        print(self.__dim)
        print(self.__no_norm)

    def fit(self, train_set, model_file=None):
        # build optional args
        options = ['-l', "%s" % self.__regularization,
                   '-k', "%s" % self.__dim,
                   '-t', "%s" % self.__iter_num,
                   '-r', "%s" % self.__learn_rate,
                   '-s', "%s" % self.__threads]
        if self.__validation_set is not None:
            options.extend(['-p', "%s" % self.__validation_set])
            if self.__auto_stop:
                options.append('--auto-stop')
        if self.__silent:
            print(self.__silent)
            options.append('--quiet')
        if self.__no_norm:
            options.append('--no-norm')
        print(options)

        args = [os.path.join(self.__libffm_path, "ffm-train")] + options + [train_set]
        if model_file:
            args += [model_file]
        # run the cmdline
        print(' '.join(args))
        subprocess.call(args, shell=False)

    def predict(self, test_set, model_file, output):
        args = [os.path.join(self.__libffm_path, "ffm-predict"),
                "%s" % test_set,
                "%s" % model_file,
                "%s" % output]
        print(' '.join(args))
        subprocess.call(args, shell=False)


myffm = Libffm()
myffm.fit('libffm_toy/criteo.tr.r100.gbdt0.ffm', 'model')
