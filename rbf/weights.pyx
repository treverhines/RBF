from __future__ import division
import numpy as np
import rbf.basis
import scipy
import scipy.spatial
import random
import logging
from itertools import combinations_with_replacement as cr
from scipy.special import binom
from functools import wraps
import mkl

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
cdef np.ndarray mvmonos(double[:,:] x,long[:,:] powers,long[:] diff):
  '''
  Description
  -----------
    multivariate monomials

  Parameters
  ----------
    x: (N,D) float array of positions where the monomials will be evaluated

    powers: (M,D) integer array of powers for each monomial

    diff: (D,) integer array of derivatives for each variable 

  Returns
  -------
    out: (M,N) float array of the differentiated monomials evaluated at x

  '''
  cdef:
    long i,j,k,l
    # number of spatial dimensions
    long D = x.shape[1]
    # number of monomials
    long M = powers.shape[0] 
    # number of positions where the monomials are evaluated
    long N = x.shape[0]
    double[:,:] out = np.empty((M,N),dtype=float)
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
          out[j,l] = coeff*x[l,i]**power              
        else:
          out[j,l] *= coeff*x[l,i]**power              

  return np.asarray(out)
  
@memoize
def monomial_powers(order,dim):
  '''
  Description
  -----------
    returns an array describing all possible monomial powers
    in a polymonial with the given order and number of
    dimensions. Calling this function with a negative order will
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
  out = []
  for p in xrange(order+1):
    if p == 0:
      out.append((0,)*dim)    
    else:
      out.extend(sum(i) for i in cr(np.eye(dim,dtype=int),p))

  return np.array(out)

  
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
  order = 0
  while (monomial_count(order+1,dim) <= stencil_size):
    order += 1

  return order  


def vpoly(nodes,order):
  '''
  Description
  -----------
    Returns the polynomial Vandermonde matrix, A[i,j], consisting of
    monomial i evaluated at node j. The monomials have a coefficient
    of 1 and powers determined from "monomial_powers"

  Paramters
  ---------
    nodes: (N,D) numpy array of points where the monomials are
      evaluated

    order: order of polynomial terms

  '''
  diff = np.zeros(nodes.shape[1],dtype=int)
  powers = monomial_powers(order,nodes.shape[1])
  out = mvmonos(nodes,powers,diff)

  return out


def dpoly(x,order,diff):
  '''
  Description
  -----------
    Returns the data vector, d[i], consisting of the differentiated
    monomial i evaluated at x. The undifferentiated monomials have a
    coefficient of 1 and powers determined from "monomial_powers"

  Parameters
  ----------
    x: (D,) numpy array where the monomials are evaluated

    order: order of polynomial terms

    diff: (D,) derivative for each spatial dimension

  '''
  x = x[None,:]
  diff = np.array(diff,dtype=int)
  powers = monomial_powers(order,x.shape[1])
  out = mvmonos(x,powers,diff)[:,0]
  
  return out


def vrbf(nodes,centers,eps,order,basis):
  '''
  Description
  -----------
    returns the matrix:

    A =   | Ar Ap.T |
          | Ap 0    |

    where Ar[i,j] is the RBF Vandermonde matrix consisting of the RBF
    with center i evaluated at nodes j.  Ap is the polynomial
    Vandermonde matrix for the indicated order.
   

  Parameters
  ----------
    nodes: (N,D) numpy array of collocation points

    centers: (N,D) numpy array of RBF centers

    eps: RBF shape parameter (constant for all RBFs)
   
    order: order of polynomial terms

    basis: callable radial basis function     

  '''
  # number of centers and dimensions
  Ns,Ndim = nodes.shape

  # number of monomial terms  
  Np = monomial_count(order,Ndim)

  # create an array of repeated eps values
  # this is faster than using np.repeat
  eps_array = np.empty(Ns)
  eps_array[:] = eps

  A = np.zeros((Ns+Np,Ns+Np))

  # Ar
  A[:Ns,:Ns] = basis(nodes,centers,eps_array).T

  # Ap
  Ap = vpoly(centers,order)  
  A[Ns:,:Ns] = Ap
  A[:Ns,Ns:] = Ap.T
  return A


def drbf(x,centers,eps,order,diff,basis): 
  '''
  Description
  -----------
    returns the vector:

      d = |dr|
          |dp|

    where dr[i] consists of a differentiated RBF with center i
    evalauted at x. dp is the polynomial data.

  '''
  #centers = np.asarray(centers)
  #x = np.asarray(x)
  x = x[None,:]

  # number of centers and dimensions
  Ns,Ndim = centers.shape

  # number of monomial terms
  Np = monomial_count(order,Ndim)

  # create an array of repeated eps values
  # this is faster than using np.repeat
  eps_array = np.empty(Ns)
  eps_array[:] = eps

  d = np.empty(Ns+Np)

  # dr
  d[:Ns] = basis(x,centers,eps_array,diff=diff)[0,:]

  # dp
  d[Ns:] = dpoly(x[0,:],order,diff)

  return d


def is_operator(diff):
  '''
  Description
  -----------
    diff can either be a tuple describing the differentiation order in
    each direction, e.g. (0,0,1), or it can describe a differential
    operator, e.g. [(1,(0,0,1)),(2,(0,1,0))], where each element
    contains a coefficient and differentiation tuple for each term in
    the operator. This function returns true if diff is describing an
    operator.

  '''
  # if each element in diff is iterable then return True. This returns
  # True if diff is an empty list 
  if all(hasattr(d,'__iter__') for d in diff):
    return True
  else:
    return False


def rbf_weight(x,nodes,diff,centers=None,
               basis=rbf.basis.phs5,order='max',
               eps=1.0):
  '''Description
  -----------
    Finds the finite difference weights, w_i, such that 

      L[f(x)] \approx \sum_i w_i*f(n_i)

    where n_i are the provided nodes and L is a differential operaotor
    specified with the diff argument.  The weights are found by
    assuming that the function, f(x), consists of a linear combination
    of M monomial terms, p_i(x), and radial basis functions, r(x),
    with N centers, c_i

      f(x) = \sum_i a_i*r(||x - c_i||) + \sum_i b_i*p_i(x)

    To ensure an equal number of equations and unknown parameters, we
    add M constaints to the parameters a_i
    
      \sum_i a_i*p_j(n_i) = 0.

    The weights are then found by solving the following linear system
 
      | r(||n_0-c_0||) ... r(||n_N-c_0||) p_0(n_0) ... p_M(n_0) |    
      |       :         :        :          :             :     | 
      | r(||n_0-c_N||) ... r(||n_N-c_N||) p_0(n_N) ... p_M(n_N) | w = d    
      |    p_0(n_0)    ...    p_0(n_N)                          |    
      |       :        ...       :                  0           |    
      |    p_M(n_0)    ...    p_M(n_N)                          |    

    where d is

          | L[r(||y-c_o||)]|_y=x |
          |         :            |
      d = | L[r(||y-c_N||)]|_y=x |
          |    L[p_0(y)]|_y=x    |
          |         :            |
          |    L[p_M(y)]|_y=x    |


  Parameters
  ----------
    x: 
    nodes:
    diff:
    centers:
    basis:
    order:
    eps:

  Note
  ----
    The overhead associated with multithreading can greatly reduce
    performance and it may be useful to set the appropriate
    environment value so that this function is run with only one
    thread.  Anaconda accelerate users can set the number of threads
    within a python script with the command mkl.set_num_threads(1)

  '''
  x = np.array(x,dtype=float,copy=True)
  nodes = np.array(nodes,dtype=float,copy=True)
  if centers is None:
    centers = nodes

  centers = np.array(centers,dtype=float,copy=True)

  if order == 'max':
    order = maximum_order(*nodes.shape)

  # center about x
  centers -= x
  nodes -= x 
  x -= x

  # number of polynomial terms that will be used
  Np = monomial_count(order,x.shape[0])
  assert Np <= nodes.shape[0], (
    'the number of monomials exceeds the number of RBFs for the '
    'stencil. Lower the polynomial order or ' 
    'increase the stencil size')

  A = vrbf(nodes,centers,eps,order,basis)
  if is_operator(diff):
    d = np.zeros(nodes.shape[0] + Np)
    for coeff,diff_tuple in diff:
      d += coeff*drbf(x,centers,eps,order,diff_tuple,basis)

  else:
    d = drbf(x,centers,eps,order,diff,basis)

  w = np.linalg.solve(A,d)[:nodes.shape[0]]

  return w 


def poly_weight(x,nodes,diff):
  '''
  finds the weights, w, such that

  f_i(c_j) = c_j**i

  | f_0(c_0) ... f_0(c_N) |     | L[f_0(y)]y=x  |
  |    :             :    | w = |     :         |
  | f_N(c_0) ... f_N(c_N) |     | L[f_N(y)]y=x  |
  '''
  x = np.array(x,copy=True)
  nodes = np.array(nodes,copy=True)
  order = rbf.weights.maximum_order(*nodes.shape)
  Np = rbf.weights.monomial_count(order,nodes.shape[1])
  assert Np == nodes.shape[0], (
    'the number of nodes in a 2D stencil needs to be 1,3,6,10,15,21,... '
    'the number of nodes in a 3D stencil needs to be 1,4,10,20,35,56,... ')
  nodes -= x
  x -= x
  A =  vpoly(nodes,order)
  d =  dpoly(x,order,diff) 
  w = np.linalg.solve(A,d)
  return w 

