import numpy as np
import rbf.utils
import unittest

class Test(unittest.TestCase):
  def test_assert_shape(self):
    A = np.random.random((10,5))
    # each of the following should be valid assertions
    rbf.utils.assert_shape(A,(10,5),'A')
    rbf.utils.assert_shape(A,(10,None),'A')
    rbf.utils.assert_shape(A,(None,5),'A')
    rbf.utils.assert_shape(A,(None,None),'A')
    # test some invalid assertions, which should raise a ValueError
    try:
      rbf.utils.assert_shape(A,(11,5),'A')
      Failed = False
    except ValueError:
      Failed = True
      
    self.assertTrue(Failed)

    try:
      rbf.utils.assert_shape(A,(10,6),'A')
      Failed = False
    except ValueError:
      Failed = True
      
    self.assertTrue(Failed)

    try:
      rbf.utils.assert_shape(A,(None,None,None),'A')
      Failed = False
    except ValueError:
      Failed = True
      
    self.assertTrue(Failed)

  def test_get_arg_count(self):
    # this function returns the number of arguments that can be
    # specified positionally
    def func1(a,b,c):
      return
    
    count = rbf.utils.get_arg_count(func1)
    self.assertTrue(count == 3)

    def func2(a,b,c=2):
      return
    
    count = rbf.utils.get_arg_count(func2)
    self.assertTrue(count == 3)

    def func3(a,b,c=2,**kwargs):
      return
    
    count = rbf.utils.get_arg_count(func3)
    self.assertTrue(count == 3)

    # this cannot be inferred and so -1 should be returned
    def func4(*args,**kwargs):
      return
    
    count = rbf.utils.get_arg_count(func4)
    self.assertTrue(count == -1)

  def test_memoize_array_input(self):
    def func(a):
      return a
    
    memfunc = rbf.utils.MemoizeArrayInput(func)  
    # call the function and make sure an entry is added to the cache
    arr = np.array([1.0])
    memfunc(np.array(arr))
    self.assertTrue(len(memfunc.cache) == 1)

    # call the 200 times, and make sure the cache only has 128 items
    # (the cache limit)
    for i in range(200):
      memfunc(np.array([i]))
    
    self.assertTrue(len(memfunc.cache) == 128)

    # clear the cache and make sure the cache size goes back to zero
    rbf.utils.clear_memoize_caches()
    self.assertTrue(len(memfunc.cache) == 0)
      

    
    
