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
def halton(long N,
           long D=1,
           long start=0,
           long skip=1,
           long prime_index=0):   
  ''' 
  Computes a halton sequence with length `N` and dimensions `D`

  Parameters
  ----------
  N : int  
    Length of the halton sequence

  D : int, optional
    Dimensions. defaults to 1

  start : int, optional
    Starting index in the halton sequence. defaults to 0
      
  skip : int, optional
    Increment by this amount. defaults to 1
      
  prime_index : int, optional
    Index of the starting prime number, defaults to 0 (i.e. 2 is the
    starting prime number).
      
  Returns
  -------
  (N, D) array

  '''
  cdef:
    long i, j
    double[:, :] seq = np.empty((N, D), dtype=float)
    long[:] p = PRIMES[prime_index:prime_index + D]

  for i in range(N):
    for j in range(D):
      seq[i, j] = halton_n(i, p[j], start, skip)

  return np.asarray(seq)


class Halton(object):
  ''' 
  Produces a Halton sequence when called and remembers the state of
  the sequence so that repeated calls produce the next items in the
  sequence

  Parameters             
  ----------         
  D : int, optional
    Dimensions, defaults to 1    

  start : int, optional
    Starting index in the Halton sequence, defaults to 0 
        
  skip : int, optional
    Increment by this amount, defaults to 1

  prime_index : int, optional
    Index of the starting prime number, defaults to 0 (i.e. 2 is the 
    starting prime number)
        
  '''
  def __init__(self, D=1, start=0, skip=1, prime_index=0):
    self.count = start
    self.skip = skip
    self.dim = D
    self.prime_index = prime_index

  def __call__(self, N):
    ''' 
    Parameters         
    ----------               
    N : int
     Number of elements to return  
                        
    Returns       
    -------        
    out : (N, D) array

    '''
    out = halton(N, self.dim, self.count, self.skip, self.prime_index)
    self.count += N*self.skip
    return out
  



