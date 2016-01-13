# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from __future__ import division
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision
from cython.parallel cimport prange


cpdef np.ndarray primes(long N):
  '''
  computes the first N prime numbers
  '''
  cdef:
    bint flag # lowered when a test number is not prime
    long test = 2 # test number
    long i,j  
    long[:] out = np.empty(N,dtype=long)

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


@cdivision(True)
cdef double halton_n(long n,
                     long base,
                     long start,
                     long skip) nogil:
  '''
  computes element n of a 1d halton sequence
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
cpdef np.ndarray halton(long N,
                        long D=1,
                        long start=0,
                        long skip=1,
                        long prime_index=0):   
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
    long i,j
    double[:,:] seq = np.empty((N,D),dtype=np.float64,order='C')
    long[:] p = primes(prime_index+D)[-D:]

  with nogil:
    for i in range(N):
      for j in range(D):
        seq[i,j] = halton_n(i,p[j],start,skip)

  return np.asarray(seq)


class Halton(object):
  '''
  A class which produces a Halton sequence when called and remembers
  the state of the sequence so that repeated calls produce the next
  items in the sequence.
  '''
  def __init__(self,D=1,start=0,skip=1,prime_index=0):
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
    self.prime_index = prime_index

  def __call__(self,N):
    '''              
    Parameters         
    ----------               
      N: Number of elements of the Halton sequence to return  
                        
    Returns       
    -------        
      (N,dim) array of elements from the Halton sequence       
    '''
    out = halton(N,self.dim,self.count,self.skip,self.prime_index)
    self.count += N*self.skip
    return out
  



