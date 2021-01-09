import sys
import inspect
import weakref
from contextlib import contextmanager
from collections import OrderedDict

import numpy as np
from scipy.spatial import cKDTree


_SHAPE_ASSERTIONS = True


def assert_shape(arr, shape, label='array'):
  '''
  Raises a ValueError if `arr` does not have the specified shape

  Parameters
  ----------
  arr : array-like

  shape : tuple
    The shape requirement for `arr`. This can have `None` to indicate that an
    axis can have any length. This can also have an Ellipsis to only enforce
    the shape for the first and/or last dimensions of `arr`

  label : str
    What to call `arr` in the error

  '''
  if not _SHAPE_ASSERTIONS:
    return

  if hasattr(arr, 'shape'):
    # if `arr` has a shape attribute then use it rather than finding the shape
    # with np.shape, which converts `arr` to a numpy array
    arr_shape = arr.shape
  else:
    arr_shape = np.shape(arr)

  arr_ndim = len(arr_shape)
  if Ellipsis in shape:
    # If `shape` has an ellipsis then only the first and/or last dimensions
    # will be checked
    start_shape = shape[:shape.index(Ellipsis)]
    end_shape = shape[shape.index(Ellipsis) + 1:]
    start_ndim = len(start_shape)
    end_ndim = len(end_shape)
    if arr_ndim < (start_ndim + end_ndim):
      raise ValueError(
        '%s is %d dimensional but it should have at least %d dimensions' %
        (label, arr_ndim, start_ndim + end_ndim))

    arr_start_shape = arr_shape[:start_ndim]
    arr_end_shape = arr_shape[arr_ndim - end_ndim:]
    for axis, (i, j) in enumerate(zip(arr_start_shape, start_shape)):
      if j is None:
        continue

      if i != j:
        raise ValueError(
          'axis %d of %s has length %d but it should have length %d'
          % (axis, label, i, j))

    for axis, (i, j) in enumerate(zip(arr_end_shape, end_shape)):
      if j is None:
        continue

      if i != j:
        raise ValueError(
          'axis %d of %s has length %d but it should have length %d'
          % (arr_ndim - end_ndim + axis, label, i, j))

  else:
    ndim = len(shape)
    if arr_ndim != ndim:
      raise ValueError(
        '%s is %d dimensional but it should have %d dimensions'
        % (label, arr_ndim, ndim))

    for axis, (i, j) in enumerate(zip(arr_shape, shape)):
      if j is None:
        continue

      if i != j:
        raise ValueError(
          'axis %d of %s has length %d but it should have length %d'
          % (axis, label, i, j))

  return


@contextmanager
def no_shape_assertions():
  '''
  Context manager that causes `assert_shape` to do nothing
  '''
  global _SHAPE_ASSERTIONS
  enter_state = _SHAPE_ASSERTIONS
  _SHAPE_ASSERTIONS = False
  yield None
  _SHAPE_ASSERTIONS = enter_state


def get_arg_count(func):
  '''
  Returns the number of arguments that can be specified positionally for a
  function. If this cannot be inferred then -1 is returned.
  '''
  # get the python version. If < 3.3 then use inspect.getargspec, otherwise use
  # inspect.signature
  if sys.version_info < (3, 3):
    argspec = inspect.getargspec(func)
    # if the function has variable positional arguments then return -1
    if argspec.varargs is not None:
      return -1

    # return the number of arguments that can be specified positionally
    out = len(argspec.args)
    return out

  else:
    params = inspect.signature(func).parameters
    # if a parameter has kind 2, then it is a variable positional argument
    if any(p.kind == 2 for p in params.values()):
      return -1

    # if a parameter has kind 0 then it is a a positional only argument and if
    # kind is 1 then it is a positional or keyword argument. Count the 0's and
    # 1's
    out = sum((p.kind == 0) | (p.kind == 1) for p in params.values())
    return out


class Memoize(object):
  '''
  An memoizing decorator. The max cache size is hard-coded at 128. When the
  limit is reached, the least recently used (LRU) item is dropped.
  '''
  # variable controlling the maximum cache size for all memoized functions
  _MAXSIZE = 128
  # collection of weak references to all instances
  _INSTANCES = []

  def __new__(cls, *args, **kwargs):
    # this keeps track of Memoize and Memoize subclass instances
    instance = object.__new__(cls)
    cls._INSTANCES += [weakref.ref(instance)]
    return instance

  def __init__(self, fin):
    self.fin = fin
    # the cache will be ordered from least to most recently used
    self.cache = OrderedDict()

  @staticmethod
  def _as_key(args):
    # convert the arguments to a hashable object. In this case, the argument
    # tuple is assumed to already be hashable
    return args

  def __call__(self, *args):
    key = self._as_key(args)
    try:
      value = self.cache[key]
      # move the item to the end signifying that it was most recently used
      try:
        # move_to_end is an attribute of OrderedDict in python 3. try calling
        # it and if the attribute does not exist then fall back to the slow
        # method
        self.cache.move_to_end(key)

      except AttributeError:
        self.cache[key] = self.cache.pop(key)

    except KeyError:
      if len(self.cache) == self._MAXSIZE:
        # remove the first item which is the least recently used item
        self.cache.popitem(0)

      value = self.fin(*args)
      # add the function output to the end of the cache
      self.cache[key] = value

    return value

  def __repr__(self):
    return self.fin.__repr__()

  def clear_cache(self):
    '''Clear the cached function output'''
    self.cache = OrderedDict()


class MemoizeArrayInput(Memoize):
  '''
  A memoizing decorator for functions that take only numpy arrays as input. The
  max cache size is hard-coded at 128. When the limit is reached, the least
  recently used (LRU) item is dropped.
  '''
  @staticmethod
  def _as_key(args):
    # create a key that is unique for the input arrays
    key = tuple((a.tobytes(), a.shape, a.dtype) for a in args)
    return key


def clear_memoize_caches():
  '''
  Clear the caches for all instances of MemoizeArrayInput
  '''
  for inst in Memoize._INSTANCES:
    if inst() is not None:
      inst().clear_cache()


class KDTree(cKDTree):
  '''
  Same as `scipy.spatial.cKDTree`, except when calling `query` with `k=1`, the
  output does not get squeezed to 1D. Also, an error will be raised if `query`
  is called with `k` larger than the number of points in the tree.
  '''
  def query(self, x, k=1, **kwargs):
    '''query the KD-tree for nearest neighbors'''
    if k > self.n:
      raise ValueError(
        'Cannot find the %s nearest points among a set of %s points'
        % (k, self.n))

    dist, indices = cKDTree.query(self, x, k=k, **kwargs)
    if k == 1:
      dist = dist[..., None]
      indices = indices[..., None]

    return dist, indices
