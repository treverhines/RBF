''' 
This module is used for evaluating the monomial basis functions which 
are commonly added to RBF interpolants
'''
from __future__ import division
from itertools import combinations_with_replacement as cr

import numpy as np
from scipy.special import binom

from rbf.utils import assert_shape, Memoize

cimport numpy as np
from cython cimport boundscheck, wraparound


def mvmonos(x, powers, diff=None):
  ''' 
  Multivariate monomial basis functions

  Parameters
  ----------
  x : (N, D) float array 
    positions where the monomials will be evaluated

  powers : (M, D) int array 
    Defines each monomial basis function using multi-index notation.  
    Each row contains the exponents for the spatial variables in a 
    monomial.

  diff : (D,) int array, optional
    derivative order for each variable

  Returns
  -------
  out : (N, M) array
    Alternant matrix where x is evaluated for each monomial 
 
  Example
  -------
  
  >>> pos = np.array([[1.0], [2.0], [3.0]])
  >>> pows = np.array([[0], [1], [2]])
  >>> mvmonos(pos,pows)
  array([[ 1.,  1.,  1.],
         [ 1.,  2.,  4.],
         [ 1.,  3.,  9.]])
           
  >>> pos = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
  >>> pows = np.array([[0, 0], [1, 0], [0, 1]])
  >>> mvmonos(pos, pows)
  array([[ 1.,  1.,  2.],
         [ 1.,  2.,  3.],
         [ 1.,  3.,  4.]])
                  
  '''
  x = np.asarray(x, dtype=float)
  assert_shape(x, (None, None), 'x')

  powers = np.asarray(powers, dtype=int)
  assert_shape(powers, (None, x.shape[1]), 'powers')
  
  if diff is None:
    diff = np.zeros(x.shape[1], dtype=int)
  else:
    diff = np.asarray(diff, dtype=int)
    
  assert_shape(diff, (x.shape[1],), 'diff') 
    
  return _mvmonos(x, powers, diff)


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray _mvmonos(double[:, :] x, 
                          long[:, :] powers, 
                          long[:] diff):
  ''' 
  cython evaluation of mvmonos
  '''
  cdef:
    long i, j, k, l
    # number of spatial dimensions
    long D = x.shape[1]
    # number of monomials
    long M = powers.shape[0] 
    # number of positions where the monomials are evaluated
    long N = x.shape[0]
    double[:, :] out = np.empty((N, M), dtype=float)
    long coeff, power

  # loop over dimensions
  for i in range(D):
    # loop over monomials
    for j in range(M):
      # find the monomial coefficients after differentiation
      coeff = 1
      for k in range(diff[i]): 
        coeff *= powers[j, i] - k

      # if the monomial coefficient is zero then make sure the power  
      # is also zero to prevent a zero division error
      if coeff == 0:
        power = 0
      else:
        power = powers[j, i] - diff[i]

      # loop over evaluation points
      for l in range(N):
        if i == 0:
          out[l, j] = coeff*x[l, i]**power              
        else:
          out[l, j] *= coeff*x[l, i]**power              

  return np.asarray(out)
  

@Memoize
def powers(order, dim):
  ''' 
  Returns an array describing the powers in all the monomial basis
  functions in a polymonial with the given order and number of
  dimensions. Calling this function with -1 for the order will return
  an empty list (no terms in the polynomial)

  Parameters
  ----------
  order : int
    Polynomial order

  dim : int
    Polynomial dimension

  Example
  -------
  This will return the powers of x and y for each monomial term in a 
  two dimensional polynomial with order 1
  
  >>> monomial_powers(1, 2) 
  >>> array([[0, 0],
             [1, 0],
             [0, 1]])
  '''
  order = int(order)
  dim = int(dim)
  
  if not (dim >= 1):
    raise ValueError('Number of dimensions must be 1 or greater')
    
  if not (order >= -1):
    raise ValueError('Polynomial order number must be -1 or greater')

  out = np.zeros((0, dim),dtype=int)
  for p in xrange(order + 1):
    if p == 0:
      outi = np.zeros((1, dim), dtype=int)
      out = np.vstack((out, outi))
    else:
      outi = np.array([sum(i) for i in cr(np.eye(dim, dtype=int), p)])
      out = np.vstack((out, outi))

  return out

  
@Memoize
def count(order, dim):
  ''' 
  Returns the number of monomial basis functions in a polynomial with 
  the given order and number of dimensions

  Parameters
  ----------
  order : int
    Polynomial order

  dim : int
    Polynomial dimension

  '''
  order = int(order)
  dim = int(dim)
  
  if not (dim >= 1):
    raise ValueError('number of dimensions must be 1 or greater')
    
  if not (order >= -1):
    raise ValueError('polynomial order number must be -1 or greater')

  return int(binom(order+dim, dim))
