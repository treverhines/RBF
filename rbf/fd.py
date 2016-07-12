''' 
This module provides functions for generating RBF-FD weights
'''
from __future__ import division
import numpy as np
import rbf.basis
import rbf.poly
import rbf.stencil
import scipy.sparse
import logging
logger = logging.getLogger(__name__)


def _default_stencil_size(nodes,dim,diff=None,diffs=None):
  max_size = nodes
  if diff is not None:
    max_order = sum(diff)

  elif diffs is not None:
    max_order = max(sum(d) for d in diffs)

  else:
    max_order = 0
    
  if max_order == 0:
    N = min(max_size,1)
    
  elif dim == 1:
    N = min(max_size,max_order + 1)    

  else:
    N = min(max_size,8)  
    
  return N


def _default_poly_order(stencil_size,dim):
  max_order = rbf.poly.maximum_order(stencil_size,dim)
  if dim == 1:
    order = max_order
  else:
    order = min(1,max_order)  

  return order


def _lhs(nodes,centers,eps,powers,basis):
  ''' 
  Returns the transposed RBF alternant matrix will added polynomial 
  terms and constraints
  '''
  # number of centers and dimensions
  Ns,Ndim = nodes.shape
  # number of monomial terms  
  Np = powers.shape[0]
  # deriviative orders
  diff = (0,)*Ndim
  A = np.zeros((Ns+Np,Ns+Np))
  A[:Ns,:Ns] = basis(nodes,centers,eps=eps,diff=diff).T
  Ap = rbf.poly.mvmonos(nodes,powers,diff=diff)
  A[Ns:,:Ns] = Ap.T
  A[:Ns,Ns:] = Ap
  return A


def _rhs(x,centers,eps,powers,diff,basis): 
  ''' 
  Returns the differentiated RBF and polynomial terms evaluated at x
  '''
  x = x[None,:]
  # number of centers and dimensions
  Ns,Ndim = centers.shape
  # number of monomial terms
  Np = powers.shape[0]
  d = np.empty(Ns+Np)
  d[:Ns] = basis(x,centers,eps,diff=diff)[0,:]
  d[Ns:] = rbf.poly.mvmonos(x,powers,diff=diff)[0,:]
  return d


def weights(x,nodes,diff=None,
            diffs=None,coeffs=None,centers=None,
            basis=rbf.basis.phs3,order=None,
            eps=None):
  ''' 
  Returns the weights which map a functions values at *nodes* to 
  estimates of that functions derivative at *x*. The weights are 
  computed using the RBF-FD method described in [1].  In this function 
  *x* is a single point in D-dimensional space. Use weight_matrix to 
  compute the weights for multiple point.

  Parameters
  ----------
    x : (D,) array
      estimation point

    nodes : (N,D) array
      observation points

    diff :(D,) int array, optional
      derivative orders for each spatial variable. This can instead be 
      specified through the diffs and coeffs arguments, which is more 
      efficient when approximating a differential operator with 
      multiple terms. If neither diff or diffs are provided then the 
      resulting weights will approximate the undifferentiated function 
      at x (i.e. diff=(0,)*D).

    diffs : (K,D) int array, optional
      derivative orders for each spatial variable for each term in a 
      differential operator. Overwrites diff if it is provided. If 
      neither diff or diffs are provided then the derivative order is 
      zero for all directions.

    coeffs : (K,) array, optional 
      list of coefficients for each derivative in diffs. does nothing 
      if diffs is not specified

    centers : (N,D) array, optional
      centers of each radial basis function. If not specified, then 
      *nodes* will be used as the centers. This is often used when 
      trying out exotic ways of imposing boundary conditions.
   
    basis : rbf.basis.RBF, optional
      radial basis function to use. Select from those available 
      in rbf.basis or create your own.
 
    order : int, optional
      Use all monomial basis functions which have an order up to and 
      including this value. Set this to 'max' to use as many monomials 
      as possible.

    eps : (N,) array, optional
      shape parameter for each radial basis function. This only makes 
      a difference when using RBFs that are not scale invariant.  All 
      the predefined RBFs except for the odd order polyharmonic 
      splines are not scale invariant.
    

  Example
  -------
    # calculate the weights for a one-dimensional second order derivative.
    >>> position = np.array([1.0]) 
    >>> nodes = np.array([[0.0],[1.0],[2.0]]) 
    >>> diff = (2,) 
    >>> weights(position,nodes,diff)

    array([ 1., -2., 1.])
    
    # calculate the weights for estimating an x derivative from three 
    # points in a two-dimensional plane
    >>> position = np.array([0.25,0.25])
    >>> nodes = np.array([[0.0,0.0],
                          [1.0,0.0],
                          [0.0,1.0]])
    >>> diff = (1,0)
    >>> weights(position,nodes,diff)

    array([ -1., 1., 0.])
    
  Note
  ----
    The overhead associated with multithreading can greatly reduce
    performance and it may be useful to set the appropriate
    environment value so that this function is run with only one
    thread.  Anaconda accelerate users can set the number of threads
    within a python script with the command mkl.set_num_threads(1)

    This function may become unstable with high order polynomials. 
    This can be somewhat remedied by shifting the coordinate system so 
    that x is zero

  References
  ----------
    [1] Fornberg, B. and N. Flyer. A Primer on Radial Basis 
        Functions with Applications to the Geosciences. SIAM, 2015.
    
  '''
  x = np.asarray(x,dtype=float)
  nodes = np.asarray(nodes,dtype=float)

  if centers is None:
    centers = nodes
  else:
    centers = np.asarray(centers,dtype=float)

  if eps is None:
    eps = np.ones(centers.shape[0])
  else:
    eps = np.asarray(eps,dtype=float)

  if order == 'max':
    order = rbf.poly.maximum_order(*nodes.shape)

  elif order is None:
    order = _default_poly_order(nodes.shape[0],nodes.shape[1])
    
  # this if else block ensures that diffs overwrites diff
  if diffs is not None:
    diffs = [tuple(d) for d in diffs]
    if coeffs is None:
      raise ValueError('coeffs must be provided along with diffs')
     
    if len(coeffs) != len(diffs):
      raise ValueError('length of coeffs must equal length of diffs')

  elif diff is not None:
    diffs = [tuple(diff)]
    coeffs = [1.0]
    
  else:
    diffs = [(0,)*x.shape[0]]
    coeffs = [1.0]
    
  powers = rbf.poly.monomial_powers(order,nodes.shape[1])
  if powers.shape[0] > nodes.shape[0]:
    raise ValueError(
      'the number of monomials exceeds the number of RBFs for the '
      'stencil. Lower the polynomial order or ' 
      'increase the stencil size')
    
  # left hand side
  lhs = _lhs(nodes,centers,eps,powers,basis)
  rhs = np.zeros(centers.shape[0] + powers.shape[0])
  for c,d in zip(coeffs,diffs):
    rhs += c*_rhs(x,centers,eps,powers,d,basis)

  try:
    out = np.linalg.solve(lhs,rhs)[:nodes.shape[0]]
  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute RBF-FD weight for point %s. Make sure that the '
       'stencil meets the conditions for non-singularity. This error '
       'may also be due to numerically flat basis functions' % x)

  return out


def poly_weights(x,nodes,diff=None,diffs=None,coeffs=None):
  ''' 
  Returns the traditional 1-D finite difference weights derived 
  from polynomial expansion. The input must have one spatial dimension.
  
  Parameters
  ----------
    x : (1,) array

    nodes : (N,1) array

    diff : (1,) int array, optional 

    diffs : (N,1) int array, optional

    coeffs : (N,) array, optional
        
  '''
  x = np.asarray(x,dtype=float)
  nodes = np.asarray(nodes,dtype=float)

  if x.shape[0] != 1:
    raise ValueError('x must have one spatial dimension to compute a poly-FD weight')

  if nodes.shape[1] != 1:
    raise ValueError('nodes must have one spatial dimension to compute a poly-FD weight')
    
  if diffs is not None:
    diffs = [tuple(d) for d in diffs]
    if coeffs is None:
      raise ValueError('coeffs must be provided along with diffs')
     
    if len(coeffs) != len(diffs):
      raise ValueError('length of coeffs must equal length of diffs')

  elif diff is not None:
    diffs = [tuple(diff)]
    coeffs = [1.0]
    
  else:
    diffs = [(0,)*x.shape[0]]
    coeffs = [1.0]

  order = rbf.poly.maximum_order(*nodes.shape)
  powers = rbf.poly.monomial_powers(order,1)
  lhs = rbf.poly.mvmonos(nodes,powers,diff=(0,)).T
  rhs = np.zeros(nodes.shape[0])
  for c,d in zip(coeffs,diffs):
    rhs += c*rbf.poly.mvmonos(x[None,:],powers,diff=d)[0,:]
  
  try:
    out = np.linalg.solve(lhs,rhs)

  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute poly-FD weight for point %s. Make sure that '
       'the stencil meets the conditions for non-singularity. ' % x)

  return out


def weight_matrix(x,nodes,diff=None,diffs=None,coeffs=None,
                  basis=rbf.basis.phs3,order=None,
                  N=None,vert=None,smp=None):
  ''' 
  Returns a weight matrix which maps a functions values at *nodes* to 
  estimates of that functions derivative at *x*.  The weight matrix is 
  made with the RBF-FD method.
  
  Parameters
  ----------
    x : (N,D) array
      estimation points

    nodes : (N,D) array
      observation points

    diff :(D,) int array, optional
      derivative orders for each spatial variable. This can instead be 
      specified through the diffs and coeffs arguments, which is more 
      efficient when approximating a differential operator with 
      multiple terms. If neither diff or diffs are provided then the 
      resulting weights will approximate the undifferentiated function 
      at x (i.e. diff=(0,)*D).

    diffs : (K,D) int array, optional
      derivative orders for each spatial variable for each term in a 
      differential operator. Overwrites diff if it is provided. If 
      neither diff or diffs are provided then the derivative order is 
      zero for all directions.

    coeffs : (K,) array, optional 
      list of coefficients for each derivative in diffs. does nothing 
      if diffs is not specified

    basis : rbf.basis.RBF, optional
      radial basis function to use. Select from those available 
      in rbf.basis or create your own.
 
    order : int, optional
      Use all monomial basis functions which have an order up to and 
      including this value. Set this to 'max' to use as many monomials 
      as possible.

    N : int, optional
      stencil size
    
    vert : (P,D) array, optional
      verticies of boundaries which stencils cannot cross
    
    smp : (Q,D) int array, optional
      connectivity of the vertices to form boundaries

  Returns
  -------
    L : (N,M) csr sparse matrix          
      
  Example
  -------
    # create a second order differentiation matrix in one-dimensional 
    # space
    >>> x = np.arange(4.0)[:,None]
    >>> W = weight_matrix(x,x,diff=(2,))
    >>> W.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])
                         
  '''
  x = np.asarray(x)
  nodes = np.asarray(nodes)
  
  if N is None:
    N = _default_stencil_size(nodes.shape[0],nodes.shape[1],
                              diff=diff,diffs=diffs)
    
  sn,dist = rbf.stencil.nearest(x,nodes,N,vert,smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)

  for i,si in enumerate(sn):
    data[i,:] = weights(x[i],nodes[si],diff=diff,
                        diffs=diffs,coeffs=coeffs,
                        basis=basis,order=order)

  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],nodes.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)

  return L
                

def diff_matrix(x,*args,**kwargs):
  ''' 
  creates a differentiation matrix which approximates a functions 
  derivative at *x* using observations of that function at *x*. The 
  weights are computed using the RBF-FD method.

  Parameters
  ----------
    x : (N,D) array
      observation points

    diff :(D,) int array, optional
      derivative orders for each spatial variable. This can instead be 
      specified through the diffs and coeffs arguments, which is more 
      efficient when approximating a differential operator with 
      multiple terms. If neither diff or diffs are provided then the 
      resulting weights will approximate the undifferentiated function 
      at x (i.e. diff=(0,)*D).

    diffs : (K,D) int array, optional
      derivative orders for each spatial variable for each term in a 
      differential operator. Overwrites diff if it is provided. If 
      neither diff or diffs are provided then the derivative order is 
      zero for all directions.

    coeffs : (K,) array, optional 
      list of coefficients for each derivative in diffs. does nothing 
      if diffs is not specified
   
    basis : rbf.basis.RBF, optional
      radial basis function to use. Select from those available 
      in rbf.basis or create your own.
 
    order : int, optional
      Use all monomial basis functions which have an order up to and 
      including this value. Set this to 'max' to use as many monomials 
      as possible.

    N : int, optional
      stencil size
    
    vert : (P,D) array, optional
      verticies of boundaries which stencils cannot cross
    
    smp : (Q,D) int array, optional
      connectivity of the vertices to form boundaries

  Returns
  -------
    L : (N,N) csr sparse matrix    
      
  Example
  -------
    # create a second order differentiation matrix in one-dimensional 
    # space
    >>> x = np.arange(4.0)[:,None]
    >>> diff_mat = diff_matrix(x,diff=(2,))
    >>> diff_mat.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])
                         
  '''
  return weight_matrix(x,x,*args,**kwargs)


def poly_diff_matrix(x,diff=None,diffs=None,coeffs=None,
                     N=None,vert=None,smp=None):
  ''' 
  creates a differentiation matrix which approximates a functions 
  derivative at *x* using observations of that function at *x*. The 
  weights are computed using the traditional finite difference method. 
  The stencil is determined by adjacency rather than nearest 
  neighbors, which results in better network connectivity.  

  Parameters
  ----------
    x : (N,D) array
      observation points
          
    diff :(D,) int array, optional
      derivative orders for each spatial variable. This can instead be 
      specified through the diffs and coeffs arguments, which is more 
      efficient when approximating a differential operator with 
      multiple terms. If neither diff or diffs are provided then the 
      resulting weights will approximate the undifferentiated function 
      at x (i.e. diff=(0,)*D).

    diffs : (K,D) int array, optional
      derivative orders for each spatial variable for each term in a 
      differential operator. Overwrites diff if it is provided. If 
      neither diff or diffs are provided then the derivative order is 
      zero for all directions.

    coeffs : (K,) array, optional 
      list of coefficients for each derivative in diffs. does nothing 
      if diffs is not specified

    N : int, optional
      stencil size
    
    vert : (P,D) array, optional
      verticies of boundaries which stencils cannot cross
    
    smp : (Q,D) int array, optional
      connectivity of the vertices to form boundaries

  Returns
  -------
    L : (N,N) csr sparse matrix    

  Example
  -------
    # create a second order differentiation matrix in one-dimensional 
    # space
    >>> x = np.arange(4.0)[:,None]
    >>> diff_mat = poly_diff_matrix(x,diff=(2,))
    >>> diff_mat.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])

  '''
  x = np.asarray(x) 

  if N is None:
    N = _default_stencil_size(x.shape[0],x.shape[1],
                              diff=diff,diffs=diffs)
    
  sn = rbf.stencil.stencil_network_1d(x,N=N,vert=vert,smp=smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = poly_weights(x[i],x[si],diff=diff,
                             diffs=diffs,coeffs=coeffs)
    
  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],x.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)

  return L


