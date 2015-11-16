# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from __future__ import division
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound
from cython.parallel cimport prange

cpdef np.ndarray primes(unsigned int N):
  '''
  computes the first N prime numbers
  '''
  cdef:
    bint flag # lowered when a test number is not prime
    unsigned int test = 2 # test number
    unsigned int i,j  
    long[:] out = np.empty(N,dtype=int)

  for i in range(N):
    while True:
      flag = True
      for j in range(i):
        if test%out[j] == 0:
          flag = False
          break      
      
      if flag:
        out[i] = test 
        break

      test += 1    

  return np.asarray(out)


cdef double halton_n(unsigned int n,
                     unsigned int base,
                     unsigned int start,
                     unsigned int skip) nogil:
  '''
  computes element n of a 1d halton sequence
  '''
  cdef:
    double out = 0
    double f = 1
    double i
  i = start + 1 + skip*n
  f = 1
  while i > 0:
    f /= base
    out += f*(i%base)
    i //= base    
     
  return out


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray halton(unsigned int N,
                        unsigned int D=1,
                        unsigned int start=0,
                        unsigned int skip=1):   
  '''
  computes a halton sequence of length N and dimensions D

  Parameters
  ----------
    N: length of the halton sequence
    D (default=1): dimensions
    start: 
    skip:

  Returns
  -------
    out: N by D array
  '''
  cdef:
    unsigned int i,j
    double[:,:] seq = np.empty((N,D),dtype=np.float64,order='C')
    long[:] p = primes(D)

  with nogil:
    for i in prange(N):
      for j in range(D):
        seq[i,j] = halton_n(i,p[j],start,skip)

  return np.asarray(seq)


class Halton(object):
  '''
  A class which produces a Halton sequence when called and remembers
  the state of the sequence so that repeated calls produce the next
  items in the sequence.
  '''
  def __init__(self,D=1,start=0,skip=1):
    '''                         
    Parameters             
    ----------         
      D (default=1): dimensions of the Halton sequence    
      start (default=0): Index to start at in the Halton sequence 
      skip (default=1): Indices to skip between successive    
        output values                            
    '''
    self.count = start
    self.skip = skip
    self.dim = D

  def __call__(self,N):
    '''              
    Parameters         
    ----------               
      N: Number of elements of the Halton sequence to return  
                        
    Returns       
    -------        
      (N,dim) array of elements from the Halton sequence       
    '''
    out = halton(N,self.dim,self.count,self.skip)
    self.count += N*self.skip
    return out
  



