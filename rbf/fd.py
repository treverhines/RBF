''' 
This module provides functions for generating RBF-FD weights
'''
from __future__ import division
import numpy as np
import rbf.basis
import rbf.poly
import rbf.stencil
import scipy.sparse
from scipy.linalg.lapack import dgetrf,dgetrs

def _lapack_solve(A,b):
  ''' 
  Solves the system of equations Ax=b, using the lapack LU routines. 
  This is faster than np.linalg.solve because it does fewer checks. A 
  and b must be double precision numpy arrays 
  '''
  lu,piv,info = dgetrf(A,overwrite_a=True)
  if info != 0:
    raise np.linalg.LinAlgError(
      'LAPACK routine *dgetrf* exited with error code %s' % info)

  x,info = dgetrs(lu,piv,b,overwrite_b=True)
  if info != 0:
    raise np.linalg.LinAlgError(
      'LAPACK routine *dgetrs* exited with error code %s' % info)

  return x

def _reshape_diffs(diffs):
  ''' 
  turns diffs into a 2D array
  '''
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
  N = rbf.poly.count(P+1,dim) - 1
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
  order = -1
  while (rbf.poly.count(order+1,dim) <= size):
    order += 1

  return order


def _lhs(s,eps,powers,basis):
  ''' 
  Returns the transposed RBF alternant matrix with added polynomial 
  terms and constraints
  '''
  # number of nodes in the stencil and the number of dimensions
  Ns,Ndim = s.shape
  # number of monomial terms  
  Np = powers.shape[0]
  # deriviative orders
  diff = np.zeros(Ndim,dtype=int)
  A = np.zeros((Ns+Np,Ns+Np),dtype=float)
  A[:Ns,:Ns] = basis(s,s,eps=eps,diff=diff).T
  Ap = rbf.poly.mvmonos(s,powers,diff=diff)
  A[Ns:,:Ns] = Ap.T
  A[:Ns,Ns:] = Ap
  return A


def _rhs(x,s,eps,powers,diff,basis): 
  ''' 
  Returns the differentiated RBF and polynomial terms evaluated at x
  '''
  x = x[None,:]
  # number of nodes in the stencil and the number of dimensions
  Ns,Ndim = s.shape
  # number of monomial terms
  Np = powers.shape[0]
  d = np.empty(Ns+Np,dtype=float)
  d[:Ns] = basis(x,s,eps,diff=diff)[0,:]
  d[Ns:] = rbf.poly.mvmonos(x,powers,diff=diff)[0,:]
  return d


def weights(x,s,diffs,coeffs=None,
            basis=rbf.basis.phs3,order=None,
            eps=None,use_pinv=False):
  ''' 
  Returns the weights which map a functions values at *s* to an 
  approximation of that functions derivative at *x*. The weights are 
  computed using the RBF-FD method described in [1]. In this function 
  *x* is a single point in D-dimensional space. Use *weight_matrix* to 
  compute the weights for multiple point.

  Parameters
  ----------
  x : (D,) array
    Target point.

  s : (N,D) array
    Source points. The interpolant used in creating the returned 
    weights will contain RBFs centered on *s*.

  diffs : (D,) int array or (K,D) int array 
    Derivative orders for each spatial variable. For example [2,0] 
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
    Type of RBF. Select from those available in rbf.basis or create 
    your own.
 
  order : int, optional
    Order of the added polynomial. This defaults to the highest 
    derivative order. For example, if *diffs* is [[2,0],[0,1]], then 
    order is set to 2. 

  eps : (N,) array, optional
    Shape parameter for each RBF, which have centers *s*. This only 
    makes a difference when using RBFs that are not scale invariant. 
    All the predefined RBFs except for the odd order polyharmonic 
    splines are not scale invariant.

  use_pinv : bool, optional
    Use the Moore-Penrose pseudo-inverse matrix to find the RBF-FD 
    weights. This should be used for stencils where the weights cannot 
    be uniquely resolved (e.g. when there are duplicate nodes).

  Returns
  -------
  out : (N,) array
    RBF-FD weights
    
  Examples
  --------
  Calculate the weights for a one-dimensional second order derivative.

  >>> x = np.array([1.0]) 
  >>> s = np.array([[0.0],[1.0],[2.0]]) 
  >>> diff = (2,) 
  >>> weights(x,s,diff)
  array([ 1., -2., 1.])
    
  Calculate the weights for estimating an x derivative from three 
  points in a two-dimensional plane

  >>> x = np.array([0.25,0.25])
  >>> s = np.array([[0.0,0.0],
                    [1.0,0.0],
                    [0.0,1.0]])
  >>> diff = (1,0)
  >>> weights(x,s,diff)
  array([ -1., 1., 0.])
    
  Notes
  -----
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
  s = np.asarray(s,dtype=float)
  diffs = np.asarray(diffs,dtype=int)
  diffs = _reshape_diffs(diffs)
  # stencil size and number of dimensions
  N,D = s.shape

  if coeffs is None:
    coeffs = np.ones(diffs.shape[0],dtype=float)
  else:
    coeffs = np.asarray(coeffs,dtype=float)
    if (coeffs.ndim != 1) | (coeffs.shape[0] != diffs.shape[0]):
      raise ValueError('coeffs and diffs have incompatible shapes')
      
  if eps is None:
    eps = np.ones(N,dtype=float)
  else:
    eps = np.asarray(eps,dtype=float)

  max_order = _max_poly_order(N,D)
  if order is None:
    order = _default_poly_order(diffs)
    order = min(order,max_order)

  if order > max_order:
    raise ValueError(
      'Polynomial order is too high for the stencil size')
    
  powers = rbf.poly.powers(order,D)
  # left hand side
  lhs = _lhs(s,eps,powers,basis)
  rhs = np.zeros(s.shape[0] + powers.shape[0],dtype=float)
  for c,d in zip(coeffs,diffs):
    rhs += c*_rhs(x,s,eps,powers,d,basis)

  if use_pinv:
    out = np.linalg.pinv(lhs).dot(rhs)[:N]

  else:  
    try:
      out = _lapack_solve(lhs,rhs)[:N]
    
    except np.linalg.LinAlgError:
      raise np.linalg.LinAlgError(
        'Cannot uniquely solve for the RBF-FD weights for point %s. '
        'Make sure that the stencil meets the conditions for '
        'non-singularity. This error may also be due to numerically '
        'flat basis functions. To ignore this error and solve for '
        'the weights with a pseudo-inversion, set *use_pinv* to True.'
        '\n\nThe stencil contains the following nodes:\n%s' % (x,s))

  return out


def weight_matrix(x,p,diffs,coeffs=None,
                  basis=rbf.basis.phs3,order=None,
                  eps=None,n=None,vert=None,smp=None,
                  use_pinv=False):
  ''' 
  Returns a weight matrix which maps a functions values at *p* to an 
  approximation of that functions derivative at *x*.  This is a 
  convenience function which first creates a stencil network and then 
  computed the RBF-FD weights for each stencil.
  
  Parameters
  ----------
  x : (N,D) array
    Target points. 

  p : (M,D) array
    Source points.

  diffs : (D,) int array or (K,D) int array 
    Derivative orders for each spatial variable. For example [2,0] 
    indicates that the weights should approximate the second 
    derivative along the x axis in two-dimensional space.  diffs can 
    also be a (K,D) array, where each (D,) sub-array is a term in a 
    differential operator. For example the two-dimensional Laplacian 
    can be represented as [[2,0],[0,2]].  

  coeffs : (K,) float array or (K,N) float, optional 
    Coefficients for each term in the differential operator specified 
    with *diffs*. Defaults to an array of ones. If *diffs* was 
    specified as a (D,) array then *coeffs* should be a length 1 
    array. If the coefficients for the differential operator vary with 
    *x* then *coeffs* can be specified as a (K,N) array.

  basis : rbf.basis.RBF, optional
    Type of RBF. Select from those available in *rbf.basis* or create 
    your own.

  order : int, optional
    Order of the added polynomial. This defaults to the highest 
    derivative order. For example, if *diffs* is [[2,0],[0,1]], then 
    *order* is set to 2. 

  eps : (M,) array, optional
    shape parameter for each RBF, which have centers *p*. This only 
    makes a difference when using RBFs that are not scale invariant.  
    All the predefined RBFs except for the odd order polyharmonic 
    splines are not scale invariant.

  n : int, optional
    Stencil size.
    
  vert : (P,D) array, optional
    Vertices of the boundary which stencils cannot intersect
   
  smp : (Q,D) int array, optional
    Connectivity of the vertices to form the boundary
    
  use_pinv : bool, optional
    Use the Moore-Penrose pseudo-inverse matrix to find the RBF-FD 
    weights. This should be used for stencils where the weights cannot 
    be uniquely resolved (e.g. when there are duplicate nodes).

  Returns
  -------
  L : (N,M) csr sparse matrix          
      
  Examples
  --------
  Create a second order differentiation matrix in one-dimensional 
  space

  >>> x = np.arange(4.0)[:,None]
  >>> W = weight_matrix(x,x,(2,))
  >>> W.toarray()
  array([[ 1., -2.,  1.,  0.],
         [ 1., -2.,  1.,  0.],
         [ 0.,  1., -2.,  1.],
         [ 0.,  1., -2.,  1.]])
                         
  '''
  x = np.asarray(x,dtype=float)
  p = np.asarray(p,dtype=float)
  diffs = np.asarray(diffs,dtype=int)
  diffs = _reshape_diffs(diffs)
  if eps is None:
    eps = np.ones(p.shape[0],dtype=float)
  
  # make *coeffs* a (K,N) array
  if coeffs is None:
    coeffs = np.ones((diffs.shape[0],x.shape[0]),dtype=float)
  else:
    coeffs = np.asarray(coeffs,dtype=float)
    if coeffs.ndim == 1:
      coeffs = np.repeat(coeffs[:,None],x.shape[0],axis=1) 
   
  if n is None:
    # if stencil size is not given then use the default stencil size. 
    # If the default stencil size is too large then incrementally 
    # decrease it
    n = _default_stencil_size(diffs)
    while True:
      try:    
        sn = rbf.stencil.stencil_network(x,p,n,vert=vert,smp=smp)
        break 

      except rbf.stencil.StencilError as err:
        n -= 1
  else:
    sn = rbf.stencil.stencil_network(x,p,n,vert=vert,smp=smp)
  
  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = weights(x[i],p[si],diffs,
                        coeffs=coeffs[:,i],eps=eps[si],
                        basis=basis,order=order,
                        use_pinv=use_pinv)

  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  shape = x.shape[0],p.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),shape)
  return L
                
