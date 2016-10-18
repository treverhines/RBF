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

def _reshape_diffs(diffs):
  ''' 
  turns diffs into a 2D array
  '''
  diffs = np.asarray(diffs)
  if diffs.ndim > 2:
    raise ValueError('diffs can only be a 1 or 2 dimensional array')
  D = diffs.shape[-1]
  out = diffs.reshape((-1,D))
  return out
  

def _default_stencil_size(diffs):
  ''' 
  returns an estimate of the number of nodes needed to do a decent job at 
  approximating the given derivative
  '''
  P = max(sum(d) for d in diffs)
  dim = len(diffs[0])
  N = rbf.poly.monomial_count(P+1,dim) - 1
  return N


def _default_poly_order(diffs):
  ''' 
  This sets the polynomial order equal to the largest derivative 
  order. So a constant and linear term (order=1) will be used when 
  approximating a first derivative. This is the smallest polynomial 
  order needed to overcome PHS stagnation error
  '''
  P = max(sum(d) for d in diffs)
  return P


def _max_poly_order(size,dim):
  ''' 
  Returns the maximum polynomial order allowed for the given stencil 
  size and number of dimensions
  '''
  if not (size >= 0):
    raise ValueError('stencil size must be 0 or greater')

  if not (dim >= 1):
    raise ValueError('number of dimensions must be 1 or greater')

  order = -1
  while (rbf.poly.monomial_count(order+1,dim) <= size):
    order += 1

  return order


def _lhs(nodes,eps,powers,basis):
  ''' 
  Returns the transposed RBF alternant matrix will added polynomial 
  terms and constraints
  '''
  # number of nodes and dimensions
  Ns,Ndim = nodes.shape
  # number of monomial terms  
  Np = powers.shape[0]
  # deriviative orders
  diff = np.zeros(Ndim,dtype=int)
  A = np.zeros((Ns+Np,Ns+Np))
  A[:Ns,:Ns] = basis(nodes,nodes,eps=eps,diff=diff).T
  Ap = rbf.poly.mvmonos(nodes,powers,diff=diff)
  A[Ns:,:Ns] = Ap.T
  A[:Ns,Ns:] = Ap
  return A


def _rhs(x,nodes,eps,powers,diff,basis): 
  ''' 
  Returns the differentiated RBF and polynomial terms evaluated at x
  '''
  x = x[None,:]
  # number of nodes and dimensions
  Ns,Ndim = nodes.shape
  # number of monomial terms
  Np = powers.shape[0]
  d = np.empty(Ns+Np)
  d[:Ns] = basis(x,nodes,eps,diff=diff)[0,:]
  d[Ns:] = rbf.poly.mvmonos(x,powers,diff=diff)[0,:]
  return d


def weights(x,nodes,diffs,coeffs=None,
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

    diffs : (D,) int array or (K,D) int array 
      derivative orders for each spatial variable. For example [2,0] 
      indicates that the weights should approximate the second 
      derivative along the x axis in two-dimensional space.  diffs can 
      also be a (K,D) array, where each (D,) sub-array is a term in a 
      differential operator. For example the two-dimensional Laplacian 
      can be represented as [[2,0],[0,2]].  

    coeffs : (K,) array, optional 
      Coefficients for each term in the differential operator 
      specified with *diffs*.  Defaults to an array of ones. If diffs 
      was specified as a (D,) array then coeffs should be a length 1 
      array.

    basis : rbf.basis.RBF, optional
      Type of radial basis function. Select from those available in 
      rbf.basis or create your own.
 
    order : int, optional
      Order of the added polynomial. This defaults to the highest 
      derivative order. For example, if *diffs* is [[2,0],[0,1]], then 
      order is set to 2. 

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
  diffs = _reshape_diffs(diffs)
  # stencil size and number of dimensions
  N,D = nodes.shape

  if coeffs is None:
    coeffs = np.ones(diffs.shape[0],dtype=float)
  else:
    coeffs = np.asarray(coeffs)
    if (coeffs.ndim != 1) | (coeffs.shape[0] != diffs.shape[0]):
      raise ValueError('coeffs and diffs have incompatible shapes')
      
  if eps is None:
    eps = np.ones(N)
  else:
    eps = np.asarray(eps,dtype=float)

  max_order = _max_poly_order(N,D)
  if order is None:
    order = _default_poly_order(diffs)
    order = min(order,max_order)

  if order > max_order:
    raise ValueError(
      'Polynomial order is too high for the stencil size')
    
  powers = rbf.poly.monomial_powers(order,D)
  # left hand side
  lhs = _lhs(nodes,eps,powers,basis)
  rhs = np.zeros(nodes.shape[0] + powers.shape[0])
  for c,d in zip(coeffs,diffs):
    rhs += c*_rhs(x,nodes,eps,powers,d,basis)

  try:
    out = np.linalg.solve(lhs,rhs)[:N]
  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'Cannot compute RBF-FD weight for point %s. Make sure that the '
       'stencil meets the conditions for non-singularity. This error '
       'may also be due to numerically flat basis functions' % x)

  return out


def poly_weights(x,nodes,diffs,coeffs=None):
  ''' 
  Returns the weights which map a functions values at *nodes* to 
  estimates of that functions derivative at *x*. The weights are 
  computed using a polynomial expansion. 
  
  For D-dimensional space The number of nodes, N, must be in the set 
  binom(P+D,D) for P in (0,1,2,...). In other words for 1-dimensional 
  space

    N in (1,2,3,4,5,6,...),
  
  for 2-dimensional space

    N in (1,3,6,10,15,21,...),

  and for 3-dimensional space

    N in (1,4,10,20,35,56,...)  
  
  Parameters
  ----------
    x : (D,) array

    nodes : (N,D) array

    diffs : (D,) int array or (K,D) int array 

    coeffs : (K,) array, optional
        
  '''
  x = np.asarray(x,dtype=float)
  nodes = np.asarray(nodes,dtype=float)
  diffs = _reshape_diffs(diffs)
  N,D = nodes.shape

  if coeffs is None:
    coeffs = np.ones(diffs.shape[0],dtype=float)
  else:
    coeffs = np.asarray(coeffs)
    if (coeffs.ndim != 1) | (coeffs.shape[0] != diffs.shape[0]):
      raise ValueError('coeffs and diffs have incompatible shapes')

  order = _max_poly_order(N,D)
  monos = rbf.poly.monomial_count(order,D)
  if N != monos:
    raise ValueError(
      'For D-dimensional space the number of nodes must be in the set '
      'binom(P+D,D) for P in (0,1,2,...)')
      
  powers = rbf.poly.monomial_powers(order,D)
  lhs = rbf.poly.mvmonos(nodes,powers).T
  rhs = np.zeros(N)
  for c,d in zip(coeffs,diffs):
    rhs += c*rbf.poly.mvmonos(x[None,:],powers,diff=d)[0,:]
  
  try:
    out = np.linalg.solve(lhs,rhs)

  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute poly-FD weight for point %s. Make sure that '
       'the stencil meets the conditions for non-singularity. ' % x)

  return out


def weight_matrix(x,nodes,diffs,coeffs=None,
                  basis=rbf.basis.phs3,order=None,
                  eps=None,size=None,vert=None,smp=None):
  ''' 
  Returns a weight matrix which maps a functions values at *nodes* to 
  estimates of that functions derivative at *x*.  The weight matrix is 
  made with the RBF-FD method.
  
  Parameters
  ----------
    x : (N,D) array
      estimation points

    nodes : (M,D) array
      observation points

    diffs : (D,) int array or (K,D) int array 
      derivative orders for each spatial variable. For example [2,0] 
      indicates that the weights should approximate the second 
      derivative along the x axis in two-dimensional space.  diffs can 
      also be a (K,D) array, where each (D,) sub-array is a term in a 
      differential operator. For example the two-dimensional Laplacian 
      can be represented as [[2,0],[0,2]].  

    coeffs : (K,) array, optional 
      Coefficients for each term in the differential operator 
      specified with *diffs*.  Defaults to an array of ones. If diffs 
      was specified as a (D,) array then coeffs should be a length 1 
      array.

    basis : rbf.basis.RBF, optional
      Type of radial basis function. Select from those available in 
      rbf.basis or create your own.
 
    order : int, optional
      Order of the added polynomial. This defaults to the highest 
      derivative order. For example, if *diffs* is [[2,0],[0,1]], then 
      order is set to 2. 

    eps : (M,) array, optional
      shape parameter for each radial basis function. This only makes 
      a difference when using RBFs that are not scale invariant.  All 
      the predefined RBFs except for the odd order polyharmonic 
      splines are not scale invariant.

    size : int, optional
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
    >>> W = weight_matrix(x,x,(2,))
    >>> W.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])
                         
  '''
  logger.debug('building RBF-FD weight matrix...')
  x = np.asarray(x)
  nodes = np.asarray(nodes)
  diffs = _reshape_diffs(diffs)
  
  if size is None:
    # if stencil size is not given then use the default stencil size. 
    # If the default stencil size is too large then incrementally 
    # decrease it
    size = _default_stencil_size(diffs)
    while True:
      try:    
        sn,dist = rbf.stencil.nearest(x,nodes,size,vert=vert,smp=smp)
        break 
      except rbf.stencil.StencilError as err:
        size -= 1
  else:
    sn,dist = rbf.stencil.nearest(x,nodes,size,vert=vert,smp=smp)
  
  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = weights(x[i],nodes[si],diffs,
                        coeffs=coeffs,eps=eps,
                        basis=basis,order=order)

  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],nodes.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)
  logger.debug('done')
  return L
                

def diff_matrix(x,diffs,coeffs=None,
                basis=rbf.basis.phs3,order=None,
                eps=None,size=None,vert=None,smp=None):
  ''' 
  creates a differentiation matrix which approximates a functions 
  derivative at *x* using observations of that function at *x*. The 
  weights are computed using the RBF-FD method.

  Parameters
  ----------
    x : (M,D) array
      observation points

    diffs : (D,) int array or (K,D) int array 
      derivative orders for each spatial variable. For example [2,0] 
      indicates that the weights should approximate the second 
      derivative along the x axis in two-dimensional space.  diffs can 
      also be a (K,D) array, where each (D,) sub-array is a term in a 
      differential operator. For example the two-dimensional Laplacian 
      can be represented as [[2,0],[0,2]].  

    coeffs : (K,) array, optional 
      Coefficients for each term in the differential operator 
      specified with *diffs*.  Defaults to an array of ones. If diffs 
      was specified as a (D,) array then coeffs should be a length 1 
      array.

    basis : rbf.basis.RBF, optional
      Type of radial basis function. Select from those available in 
      rbf.basis or create your own.
 
    order : int, optional
      Order of the added polynomial. This defaults to the highest 
      derivative order. For example, if *diffs* is [[2,0],[0,1]], then 
      order is set to 2. 

    eps : (M,) array, optional
      shape parameter for each radial basis function. This only makes 
      a difference when using RBFs that are not scale invariant.  All 
      the predefined RBFs except for the odd order polyharmonic 
      splines are not scale invariant.

    size : int, optional
      stencil size
    
    vert : (P,D) array, optional
      verticies of boundaries which stencils cannot cross
    
    smp : (Q,D) int array, optional
      connectivity of the vertices to form boundaries

  Returns
  -------
    L : (M,M) csr sparse matrix    
      
  Example
  -------
    # create a second order differentiation matrix in one-dimensional 
    # space
    >>> x = np.arange(4.0)[:,None]
    >>> diff_mat = diff_matrix(x,(2,))
    >>> diff_mat.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])
                         
  '''
  logger.debug('building RBF-FD differentiation matrix...')
  x = np.asarray(x)
  diffs = _reshape_diffs(diffs)

  if size is None:
    # if stencil size is not given then use the default stencil size. 
    # If the default stencil size is too large then incrementally 
    # decrease it
    size = _default_stencil_size(diffs)
    while True:
      try:    
        sn = rbf.stencil.stencil_network(x,size,vert=vert,smp=smp)
        break 
      except rbf.stencil.StencilError as err:
        size -= 1
  else:
    sn = rbf.stencil.stencil_network(x,size,vert=vert,smp=smp)
    
  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = weights(x[i],x[si],diffs,
                        coeffs=coeffs,eps=eps,
                        basis=basis,order=order)

  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],x.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)
  logger.debug('done')
  return L


def diff_matrix_1d(x,diffs,coeffs=None,
                   size=None,vert=None,smp=None):
  ''' 
  creates a differentiation matrix which approximates a functions 
  derivative at *x* using observations of that function at *x*. The 
  weights are computed using the traditional finite difference method. 
  The stencil is determined by adjacency rather than nearest 
  neighbors, which results in better network connectivity.  

  Parameters
  ----------
    x : (M,D) array
      observation points
          
    diffs : (D,) int array or (K,D) int array 
      derivative orders for each spatial variable. For example [2,0] 
      indicates that the weights should approximate the second 
      derivative along the x axis in two-dimensional space.  diffs can 
      also be a (K,D) array, where each (D,) sub-array is a term in a 
      differential operator. For example the two-dimensional Laplacian 
      can be represented as [[2,0],[0,2]].  

    coeffs : (K,) array, optional 
      Coefficients for each term in the differential operator 
      specified with *diffs*.  Defaults to an array of ones. If diffs 
      was specified as a (D,) array then coeffs should be a length 1 
      array.

    size : int, optional
      stencil size. If N neighbors cannot be found, then incrementally 
      smaller values of N are tested until a stencil network can be 
      formed.
    
    vert : (P,D) array, optional
      verticies of boundaries which stencils cannot cross
    
    smp : (Q,D) int array, optional
      connectivity of the vertices to form boundaries

  Returns
  -------
    L : (M,M) csr sparse matrix    

  Example
  -------
    # create a second order differentiation matrix in one-dimensional 
    # space
    >>> x = np.arange(4.0)[:,None]
    >>> diff_mat = poly_diff_matrix(x,(2,))
    >>> diff_mat.toarray()

    array([[ 1., -2.,  1.,  0.],
           [ 1., -2.,  1.,  0.],
           [ 0.,  1., -2.,  1.],
           [ 0.,  1., -2.,  1.]])

  '''
  logger.debug('building polynomial differentiation matrix...')
  x = np.asarray(x) 
  diffs = _reshape_diffs(diffs)

  if size is None:
    # if stencil size is not given then use the default stencil size. 
    # If the default stencil size is too large then incrementally 
    # decrease it
    size = _default_stencil_size(diffs)
    while True:
      try:    
        sn = rbf.stencil.stencil_network_1d(x,size,vert=vert,smp=smp)
        break 
      except rbf.stencil.StencilError as err:
        size -= 1
  else:
    sn = rbf.stencil.stencil_network_1d(x,size,vert=vert,smp=smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = poly_weights(x[i],x[si],diffs,coeffs=coeffs)
    
  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],x.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)
  logger.debug('done')
  return L


