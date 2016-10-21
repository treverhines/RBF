''' 
This module defines a function and class for generating halton 
sequences
'''
from __future__ import division
import numpy as np
cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision

@cdivision(True)
@boundscheck(False)
cpdef np.ndarray primes(long N):
  ''' 
  computes the first N prime numbers
  '''
  cdef:
    bint flag # lowered when a test number is not prime
    long test = 2 # test number
    long i,j  
    long[:] out = np.empty(N,dtype=np.int)

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


PRIMES = primes(1000)

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
    N : int  
      length of the halton sequence

    D : int, optional
      dimensions. defaults to 1

    start : int, optional
      starting index in the halton sequence. defaults to 0
      
    skip: int, optional
      increment by this amount. defaults to 1
      
    prime_index : int, optional
      index of the starting prime number, defaults to 0 (i.e. 2 is the 
      starting prime number)
      
  Returns
  -------
    out : (N,D) array

  '''
  cdef:
    long i,j
    double[:,:] seq = np.empty((N,D),dtype=np.float64)
    long[:] p = PRIMES[prime_index:prime_index+D]

  with nogil:
    for i in range(N):
      for j in range(D):
        seq[i,j] = halton_n(i,p[j],start,skip)

  return np.asarray(seq)


class Halton(object):
  ''' 
  Produces a Halton sequence when called and remembers the state of 
  the sequence so that repeated calls produce the next items in the 
  sequence
  '''
  def __init__(self,D=1,start=0,skip=1,prime_index=0):
    ''' 
    Parameters             
    ----------         
      D : int, optional
        dimensions, defaults to 1    

      start : int, optional
        starting index in the Halton sequence, defaults to 0 
        
      skip : int, optional
        increment by this amount, defaults to 1

      prime_index : int, optional
        index of the starting prime number, defaults to 0 (i.e. 2 is the 
        starting prime number)
        
    '''
    self.count = start
    self.skip = skip
    self.dim = D
    self.prime_index = prime_index

  def __call__(self,N):
    ''' 
    Parameters         
    ----------               
      N : int
       number of elements of the Halton sequence to return  
                        
    Returns       
    -------        
      out : (N,D) array
    '''
    out = halton(N,self.dim,self.count,self.skip,self.prime_index)
    self.count += N*self.skip
    return out
  



