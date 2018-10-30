import sys
import inspect
import weakref
from collections import OrderedDict

import numpy as np


def assert_shape(arr, shape, label):
  ''' 
  Raises an error if `arr` does not have the specified shape. If an
  element in `shape` is `None` then that axis can have any length.
  '''
  if hasattr(arr, 'shape'):
    arr_shape = arr.shape
  else:
    arr_shape = np.shape(arr)
    
  if len(arr_shape) != len(shape):
    raise ValueError(
      '`%s` is a %s dimensional array but it should be a %s '
      'dimensional array' % (label, len(arr_shape), len(shape)))

  for axis, (i, j) in enumerate(zip(arr_shape, shape)):
    if j is None:
      continue

    if i != j:
      raise ValueError(
        'Axis %s of `%s` has length %s but it should have length '
        '%s.' % (axis, label, i, j))

  return


def get_arg_count(func):
  ''' 
  Returns the number of arguments that can be specified positionally
  for a function. If this cannot be inferred then -1 is returned.
  '''
  # get the python version. If < 3.3 then use inspect.getargspec,
  # otherwise use inspect.signature
  if sys.version_info < (3, 3):
    argspec = inspect.getargspec(func)
    # if the function has variable positional arguments then return -1
    if argspec.varargs is not None:
      return -1

    # return the number of arguments that can be specified
    # positionally
    out = len(argspec.args)
    return out
  
  else:  
    params = inspect.signature(func).parameters
    # if a parameter has kind 2, then it is a variable positional
    # argument
    if any(p.kind == 2 for p in params.values()):
      return -1

    # if a parameter has kind 0 then it is a a positional only
    # argument and if kind is 1 then it is a positional or keyword
    # argument. Count the 0's and 1's
    out = sum((p.kind == 0) | (p.kind == 1) for p in params.values())
    return out


class MemoizeArrayInput:
    '''
    A memoizing decorator for functions that take only numpy arrays as
    input. The max cache size is hard-coded at 128. When the limit is
    reached, the least recently used (LRU) item is dropped.
    
    Parameters
    ----------
    fin : function
        Function that takes arrays as input
    
    '''
    # variable controlling the maximum cache size for all memoized
    # functions
    _MAXSIZE = 128
    # collection of weak references to all instances
    _INSTANCES = []

    def __init__(self, fin):
        self.fin = fin
        # the cache is ordered from least to most recently used
        self.cache = OrderedDict()
        MemoizeArrayInput._INSTANCES += [weakref.ref(self)]    

    def __call__(self, *args):
        # create a key that is unique for the input arrays
        key = tuple((a.tobytes(), a.shape, a.dtype) for a in args)
        try:
            value = self.cache[key]
            # move this key to the end signifying that is was most
            # recently used
            self.cache.move_to_end(key)

        except KeyError:
            if len(self.cache) == MemoizeArrayInput._MAXSIZE:
                # remove the first item which is the least recently
                # used item
                self.cache.popitem(0)
            
            value = self.fin(*args)
            # add the function output to the end of the cache
            self.cache[key] = value

        return value
             
    def __repr__(self):
        return self.fin.__repr__()

    def clear_cache(self):
        self.cache = OrderedDict()


def clear_memoize_array_input_caches():
    '''
    Clear the caches for all instances of MemoizeArrayInput
    ''' 
    for inst in MemoizeArrayInput._INSTANCES:
        if inst() is not None:
            inst().clear_cache()
