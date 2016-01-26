from __future__ import division
import numpy as np
import rbf.basis
import logging
from itertools import combinations_with_replacement as cr
from scipy.special import binom
from functools import wraps

cimport numpy as np
from cython cimport boundscheck,wraparound,cdivision

logger = logging.getLogger(__name__)

def memoize(f):
  '''
  Description
  -----------
    decorator that stores the output of functions with hashable
    arguments and returns that output when the function is called
    again with the same arguments.

  Note
  ----
    Cached output is not copied. If the function output is mutable
    then any modifications to the output will result in modifications
    to the cached output

  '''
  cache = {}
  @wraps(f)
  def fout(*args):
    if args not in cache:
      cache[args] = f(*args)
    return cache[args]

  return fout  


@boundscheck(False)
@wraparound(False)
cpdef np.ndarray mvmonos(double[:,:] x,long[:,:] powers,long[:] diff):
  '''
  Description
  -----------
    multivariate monomials

  Parameters
  ----------
    x: (N,D) float array of positions where the monomials will be
      evaluated

    powers: (M,D) integer array of powers for each monomial

    diff: (D,) integer array of derivatives for each variable 

  Returns
  -------
    out: (N,M) Alternant matrix where x is evaluated for each monomial
      term

  Note
  ----
    This is a Cython function and all the input must be numpy arrays

  '''
  cdef:
    long i,j,k,l
    # number of spatial dimensions
    long D = x.shape[1]
    # number of monomials
    long M = powers.shape[0] 
    # number of positions where the monomials are evaluated
    long N = x.shape[0]
    double[:,:] out = np.empty((N,M),dtype=float)
    long coeff,power

  # loop over dimensions
  for i in range(D):
    # loop over monomials
    for j in range(M):
      # find the monomial coefficients after differentiation
      coeff = 1
      for k in range(diff[i]): 
        coeff *= powers[j,i] - k

      # if the monomial coefficient is zero then make sure the power  
      # is also zero to prevent a zero division error
      if coeff == 0:
        power = 0
      else:
        power = powers[j,i] - diff[i]

      # loop over evaluation points
      for l in range(N):
        if i == 0:
          out[l,j] = coeff*x[l,i]**power              
        else:
          out[l,j] *= coeff*x[l,i]**power              

  return np.asarray(out)
  
@memoize
def monomial_powers(order,dim):
  '''
  Description
  -----------
    returns an array describing all possible monomial powers
    in a polymonial with the given order and number of
    dimensions. Calling this function with -1 for the order will
    return an empty list (no terms in the polynomial)

  Parameters
  ----------
    order: polynomial order

    dim: polynomial dimension

  Example
  -------
    This will return the powers of x and y for each monomial term in a
    two dimensional polynomial with order 1 

      In [1]: monomial_powers(1,2) 
      Out[1]: array([[0,0],
                     [1,0],
                     [0,1]])
  '''
  assert dim >= 1, 'number of dimensions must be 1 or greater'
  assert order >= -1, 'polynomial order number must be -1 or greater'

  out = np.zeros((0,dim),dtype=int)
  for p in xrange(order+1):
    if p == 0:
      outi = np.zeros((1,dim),dtype=int)
      out = np.vstack((out,outi))
    else:
      outi = np.array([sum(i) for i in cr(np.eye(dim,dtype=int),p)])
      out = np.vstack((out,outi))

  return out

  
@memoize
def monomial_count(order,dim):
  '''
  Description
  -----------
    returns the number of monomial terms in a polynomial with the
    given order and number of dimensions

  Parameters
  ----------
    order: polynomial order

    dim: polynomial dimension

  '''
  assert dim >= 1, 'number of dimensions must be 1 or greater'
  assert order >= -1, 'polynomial order number must be -1 or greater'

  return int(binom(order+dim,dim))


@memoize
def maximum_order(stencil_size,dim):
  '''
  Description
  -----------
    returns the maximum polynomial order allowed for the given stencil
    size and number of dimensions

  Parameters
  ----------
    stencil_size: number of nodes in the stencil

    dim: spatial dimensions of the stencil

  '''
  assert stencil_size >= 0, 'stencil size must be 0 or greater'
  assert dim >= 1, 'number of dimensions must be 1 or greater'

  order = -1
  while (monomial_count(order+1,dim) <= stencil_size):
    order += 1

  return order  

