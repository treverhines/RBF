''' 
This module defines a function and class for generating halton 
sequences
'''
from __future__ import division
import numpy as np

cimport numpy as np
from cython cimport boundscheck, wraparound, cdivision

# first 100 prime numbers
PRIMES = np.array([  2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,
                    43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
                   103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                   173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
                   241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                   317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                   401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                   479, 487, 491, 499, 503, 509, 521, 523, 541])


@cdivision(True)
cdef double halton_n(long n,
                     long base,
                     long start,
                     long skip) nogil:
  ''' 
  Computes element n of a 1d halton sequence
  '''
  cdef:
    double out = 0
    double f = 1
    long i

  i = start + 1 + skip*n
  while i > 0:
    f /= base
    out += f*(i%base)
    i //= base    
     
  return out


@boundscheck(False)
@wraparound(False)
def halton_sequence(long size,
                    long dim=1,
                    long start=100,
                    long skip=1,
                    long prime_index=0):   
  ''' 
  Returns a Halton sequence with length `size` and dimensions `dim`

  Parameters
  ----------
  size : int  
    Number of Halton sequence elements to return

  dim : int, optional
    Number of dimensions. defaults to 1

  start : int, optional
    Starting index for the Halton sequence. defaults to 100
      
  skip : int, optional
    Increment by this amount. defaults to 1
      
  prime_index : int, optional
    Index of the starting prime number, defaults to 0 (i.e. 2 is the
    starting prime number).
      
  Returns
  -------
  (size, dim) array

  '''
  cdef:
    long i, j
    double[:, :] seq = np.empty((size, dim), dtype=float)
    long[:] p = PRIMES[prime_index:prime_index + dim]

  for i in range(size):
    for j in range(dim):
      seq[i, j] = halton_n(i, p[j], start, skip)

  return np.asarray(seq)


class HaltonSequence(object):
  ''' 
  Produces a Halton sequence when called and remembers the state of
  the sequence so that repeated calls produce the next items in the
  sequence

  Parameters             
  ----------         
  dim : int, optional
    Dimensions, defaults to 1    

  start : int, optional
    Starting index in the Halton sequence, defaults to 100 
        
  skip : int, optional
    Increment by this amount, defaults to 1

  prime_index : int, optional
    Index of the starting prime number, defaults to 0 (i.e. 2 is the 
    starting prime number)
        
  '''
  def __init__(self, dim=1, start=100, skip=1, prime_index=0):
    self.count = start
    self.skip = skip
    self.dim = dim
    self.prime_index = prime_index

  def __call__(self, size=None):
    return self.random(size=size)

  def random(self, size=None):
    ''' 
    Returns elements of a Halton sequence with values between 0 and 1
    
    Parameters         
    ----------               
    size : int, optional
      Number of elements to return. If this is not given, then a
      single element will be returned as a (dim,) array.
                        
    Returns       
    -------        
    out : (dim,) or (size, dim) array

    '''
    if size is None:
        size = 1
        output_1d = True

    else:
        output_1d = False
            
    out = halton_sequence(size, 
                          self.dim, 
                          self.count, 
                          self.skip, 
                          self.prime_index)
    self.count += size*self.skip

    if output_1d:
        out = out[0]
    
    return out
  
  def randint(self, a, b=None, size=None):
    '''
    Returns elements of the Halton sequence that have been mapped to
    integers between `a` and `b`. If `b` is not given, then the
    returned values are between 0 and `a`.

    Parameters
    ----------
    a : int

    b : int, optional
    
    '''
    if b is None:
        b = a
        a = 0
    
    a, b = int(a), int(b)
    out = self.random(size=size)
    out = (out*(b - a) + a).astype(int)
    return out
        
  def uniform(self, low=0.0, high=1.0, size=None):
    '''
    Returns elements of the Halton sequence that have been linearly
    mapped to floats between `low` and `high`.

    Parameters
    ----------
    low : float or (dim,) float array

    high : float or (dim,) float array
    
    '''
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    out = self.random(size=size)
    out = (out*(high - low) + low)
    return out
  

class GlobalHaltonSequence(HaltonSequence):
    '''
    A Halton sequence whose `count` attribute is an alias for the class-level
    variable `COUNT`. This ensures that two instances of a GlobalHaltonSequence
    will produce different sequences.
    '''
    # this variable is shared among 
    COUNT = 100
    
    @property
    def count(self):
        return GlobalHaltonSequence.COUNT

    @count.setter
    def count(self, val):
        GlobalHaltonSequence.COUNT = val
        
    def __init__(self, dim=1, skip=1, prime_index=0):
        HaltonSequence.__init__(
            self, 
            dim=dim, 
            start=GlobalHaltonSequence.COUNT, 
            skip=skip, 
            prime_index=prime_index)
