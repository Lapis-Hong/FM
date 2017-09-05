# coding: utf-8
from __future__ import print_function
import time
from functools import wraps
try:
    import cPickle as pickle
except ImportError:
    import pickle


# if __all__ then the above import statement can't be import from other module by from data_process import *
# __all__ = ['clock', 'keyword_only']

# parametric decorator
def clock(info=''):  # generator a decorator
    def clocked(func):  # the true decorator function
        @wraps(func)
        def wrapper(*args, **kwargs):  # passing params to func
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            print(info + '  Call ' + func.__name__ + '() Take {0} sec.\n'.format(elapsed))
            # in python 2.6, format need to specify the location like {0}, {1}
            return result  # important
        return wrapper
    return clocked


def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 1:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        wrapper._input_kwargs = kwargs
        return func(*args, **kwargs)
    return wrapper
