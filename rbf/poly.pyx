''' 
This module is used for evaluating the monomial basis functions which 
are commonly added to RBF interpolants
'''

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
  decorator that stores the output of functions with hashable 
  arguments and returns that output when the function is called again 
  with the same arguments.

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


def mvmonos(x,powers,diff=None):
  ''' 
  Multivariate monomial basis functions

  Parameters
  ----------
    x : (N,D) float array 
      positions where the monomials will be evaluated

    powers : (M,D) int array 
      Defines each monomial basis function using multi-index notation.  
      Each row contains the exponents for each spatial variable in a 
      monomial.

    diff : (D,) int array, optional
      derivative order for each variable

  Returns
  -------
    out: (N,M) Alternant matrix where x is evaluated for each monomial 
 
  Example
  -------
    # compute f1(x) = 1, f2(x) = x, f3(x) = x**2 at positions 1.0, 
    # 2.0, and 3.0
    >>> pos = np.array([[1.0],[2.0],[3.0]])
    >>> pows = np.array([[1],[2],[3]])
    >>> mvmonos(pos,pows)

    array([[ 1.,  1.,  1.],
           [ 1.,  2.,  4.],
           [ 1.,  3.,  9.]])
           
    # compute f1(x,y) = 1, f2(x,y) = x, f3(x,y) = y at positions 
    # [1.0,2.0], [2.0,3.0], and [3.0,4.0]
    >>> pos = np.array([[1.0,2.0],[2.0,3.0],[3.0,4.0]])
    >>> pows = np.array([[0,0],[1,0],[0,1]])
    >>> mvmonos(pos,pows)

    array([[ 1.,  1.,  2.],
           [ 1.,  2.,  3.],
           [ 1.,  3.,  4.]])
                  
  '''
  x = np.asarray(x,dtype=float)
  powers = np.asarray(powers,dtype=int)
  if diff is None:
    diff = (0,)*x.shape[1]
  else:
    diff = tuple(diff)

  return _mvmonos(x,powers,diff)


@boundscheck(False)
@wraparound(False)
cdef np.ndarray _mvmonos(double[:,:] x,long[:,:] powers,tuple diff):
  ''' 
  cython evaluation of mvmonos
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
  Returns an array describing all the monomial basis functions in a 
  polymonial with the given order and number of dimensions. Calling 
  this function with -1 for the order will return an empty list (no 
  terms in the polynomial)

  Parameters
  ----------
    order : int
      polynomial order

    dim : int
      polynomial dimension

  Example
  -------
    # This will return the powers of x and y for each monomial term in a
    # two dimensional polynomial with order 1 
    >>> monomial_powers(1,2) 
    >>> array([[0,0],
               [1,0],
               [0,1]])
  '''
  if not (dim >= 1):
    raise ValueError('number of dimensions must be 1 or greater')
    
  if not (order >= -1):
    raise ValueError('polynomial order number must be -1 or greater')

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
  Returns the number of monomial basis functions in a polynomial with 
  the given order and number of dimensions

  Parameters
  ----------
    order : int
      polynomial order

    dim : int
      polynomial dimension

  '''
  if not (dim >= 1):
    raise ValueError('number of dimensions must be 1 or greater')
    
  if not (order >= -1):
    raise ValueError('polynomial order number must be -1 or greater')

  return int(binom(order+dim,dim))


@memoize
def maximum_order(stencil_size,dim):
  ''' 
  Returns the maximum polynomial order allowed for the given stencil 
  size and number of dimensions

  Parameters
  ----------
    stencil_size : int
      number of nodes in the stencil

    dim : int
      spatial dimensions

  '''
  if not (stencil_size >= 0):
    raise ValueError('stencil size must be 0 or greater')
    
  if not (dim >= 1):
    raise ValueError('number of dimensions must be 1 or greater')

  order = -1
  while (monomial_count(order+1,dim) <= stencil_size):
    order += 1

  return order  

