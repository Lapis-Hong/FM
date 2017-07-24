from __future__ import print_function
import time
from functools import wraps

__all__ = ['embeddingIndexInit', 'fm_index']


# parametric decorator
INFO = 'Take {}.'
def clock(info = INFO):  # generator a decorator
    def decorate(func):  # the true decorator function
        @wraps(func)
        def clocked(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            print(info.format(elapsed))
            return result
        return clocked
    return decorate
