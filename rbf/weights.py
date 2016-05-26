from __future__ import division
import numpy as np
import rbf.basis
import rbf.poly
import logging
from itertools import combinations_with_replacement as cr
from scipy.special import binom
from functools import wraps

logger = logging.getLogger(__name__)


def apoly(nodes,order):
  ''' 
  Description
  -----------
    Returns the polynomial alternant matrix where each monomial is
    evaluated at each node. The monomials have a coefficient of 1 and
    consist of all those that would be in a polynomial with the given
    order. The returned alternant matrix is the transpose of the
    standard alternant matrix.

  Parameters
  ---------
    nodes: (N,D) numpy array of points where the monomials are
      evaluated

    order: polynomial order

  '''
  diff = np.zeros(nodes.shape[1],dtype=int)
  powers = rbf.poly.monomial_powers(order,nodes.shape[1])
  out = rbf.poly.mvmonos(nodes,powers,diff).T

  return out


def dpoly(x,order,diff):
  ''' 
  Description
  -----------
    Returns the data vector consisting of the each differentiated
    monomial evaluated at x. The undifferentiated monomials have a
    coefficient of 1 and powers determined from "monomial_powers"

  Parameters
  ----------
    x: (D,) numpy array where the monomials are evaluated

    order: order of polynomial terms

    diff: (D,) derivative for each spatial dimension

  '''
  x = x[None,:]
  diff = np.array(diff,dtype=int)
  powers = rbf.poly.monomial_powers(order,x.shape[1])
  out = rbf.poly.mvmonos(x,powers,diff)[0,:]
  
  return out


def arbf(nodes,centers,eps,order,basis):
  ''' 
  Description
  -----------
    returns the matrix:

    A =   | Ar Ap.T |
          | Ap 0    |

    where Ar is the transposed RBF alternant matrix. And Ap is the
    transposed polynomial alternant matrix.

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
  Np = rbf.poly.monomial_count(order,Ndim)

  # create an array of repeated eps values
  # this is faster than using np.repeat
  eps_array = np.empty(Ns)
  eps_array[:] = eps

  A = np.zeros((Ns+Np,Ns+Np))

  # Ar
  A[:Ns,:Ns] = basis(nodes,centers,eps_array).T

  # Ap
  Ap = apoly(centers,order)  
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


    where dr consists of the differentiated RBFs evalauted at x and dp
    consists of the monomials evaluated at x

  '''
  x = x[None,:]

  # number of centers and dimensions
  Ns,Ndim = centers.shape

  # number of monomial terms
  Np = rbf.poly.monomial_count(order,Ndim)

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
  ''' 
  Description
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

    This function shifts the coordinate system so that x is at the
    origin.  This makes the added monomials equivalent to the terms in
    a Taylor series expansion about x.

  '''
  x = np.array(x,dtype=float,copy=True)
  nodes = np.array(nodes,dtype=float,copy=True)
  if centers is None:
    centers = nodes

  centers = np.array(centers,dtype=float,copy=True)

  if order == 'max':
    order = rbf.poly.maximum_order(*nodes.shape)

  # center about x
  centers -= x
  nodes -= x 
  x -= x

  # number of polynomial terms that will be used
  Np = rbf.poly.monomial_count(order,x.shape[0])
  assert Np <= nodes.shape[0], (
    'the number of monomials exceeds the number of RBFs for the '
    'stencil. Lower the polynomial order or ' 
    'increase the stencil size')

  A = arbf(nodes,centers,eps,order,basis)
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
  order = rbf.poly.maximum_order(*nodes.shape)
  Np = rbf.poly.monomial_count(order,nodes.shape[1])
  assert Np == nodes.shape[0], (
    'the number of nodes in a 2D stencil needs to be 1,3,6,10,15,21,... '
    'the number of nodes in a 3D stencil needs to be 1,4,10,20,35,56,... ')
  nodes -= x
  x -= x
  A =  apoly(nodes,order)
  d =  dpoly(x,order,diff) 
  w = np.linalg.solve(A,d)
  return w 

