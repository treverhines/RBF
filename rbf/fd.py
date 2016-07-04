from __future__ import division
import numpy as np
import rbf.basis
import rbf.poly
import rbf.stencil
import scipy.sparse
import logging

logger = logging.getLogger(__name__)


def _arbf(nodes,centers,eps,powers,basis):
  ''' 
  Returns the matrix:

  A =   | Ar Ap.T |
        | Ap 0    |

  where Ar is the transposed RBF alternant matrix. And Ap is the
  transposed polynomial alternant matrix.

  Parameters
  ----------
    nodes : (N,D) float array 
      collocation points

    centers: (N,D) float array
      RBF centers

    eps : (N,) float array
      RBF shape parameter
   
    powers : (M,D) int array
      order of polynomial terms

    basis : rbf.basis.RBF instance

  '''
  # number of centers and dimensions
  Ns,Ndim = nodes.shape

  # number of monomial terms  
  Np = len(powers)

  # deriviative orders
  diff = (0,)*Ndim
  
  A = np.zeros((Ns+Np,Ns+Np))
  A[:Ns,:Ns] = basis(nodes,centers,eps=eps,diff=diff,check_input=False).T
  Ap = rbf.poly.mvmonos(nodes,powers,diff=diff,check_input=False).T
  A[Ns:,:Ns] = Ap
  A[:Ns,Ns:] = Ap.T
  return A


def _drbf(x,centers,eps,powers,diff,basis): 
  ''' 
  returns the vector:

    d = |dr|
        |dp|


  where dr consists of the differentiated RBFs evalauted at x and dp
  consists of the monomials evaluated at x

  Parameters
  ----------
    x : (D,) float array 
      collocation points

    centers: (N,D) float array
      RBF centers

    eps : (N,) float array
      RBF shape parameter
   
    powers : (M,D) int array
      order of polynomial terms

    diff : (D,) int tuple
      derivative orders
      
    basis : rbf.basis.RBF instance
  '''
  x = x[None,:]

  # number of centers and dimensions
  Ns,Ndim = centers.shape

  # number of monomial terms
  Np = len(powers)

  d = np.empty(Ns+Np)
  d[:Ns] = basis(x,centers,eps,diff=diff,check_input=False)[0,:]
  d[Ns:] = rbf.poly.mvmonos(x,powers,diff=diff,check_input=False)[0,:]

  return d


def diff_weights(x,nodes,diff=None,
                 diffs=None,coeffs=None,centers=None,
                 basis=rbf.basis.phs3,order=None,
                 eps=1.0):
  ''' 
  computes the weights used for a finite difference approximation at x.
  The weights are computed using the RBF-FD method described in [1].

  Parameters
  ----------
    x : (D,) array
      position where the derivative is being approximated

    nodes : (N,D) array
      nodes adjacent to x

    diff :(D,) int tuple, may specify diffs and coeffs instead
      derivative orders for each spatial dimension. 

    centers : (N,D) array, optional
      centers of each radial basis function. If not specified, then 
      the nodes will be used as centers. This is often used when 
      trying out exotic ways of imposing boundary conditions.
   
    basis : rbf.basis.RBF, optional
      radial basis function to use. Select from those available 
      in rbf.basis
 
    order : int, optional
      order of added polynomial terms.  can be 'max' to use the 
      largest number of polynomials without creating a singular 
      matrix.  This may lead to lead to instabilities in derivative 
      approximations. 1 is generally a safe value

    eps : float, optional
      shape parameter. This only makes a difference when using RBFs 
      that are not scale invariant, which you should not do. Any of 
      the odd PHS basis function are unaffected by the shape 
      parameter. However, if the problem is particularly poorly scaled 
      then eps may be a good way to scale the problem to something 
      sensible.
    
    diffs : (K,) list of (D,) int tuples, optional
      derivative terms. if specified then it overwrites diff and coeffs 
      must also be specified

    coeffs : (K,) array, optional 
      list of coefficients for each derivative in diffs. does nothing 
      if diffs is not specified

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
    [1] B. Fornberg and N. Flyer. A Primer on Radial Basis 
        Functions with Applications to the Geosciences. SIAM, 2015.
    
  '''
  x = np.asarray(x,dtype=float)
  nodes = np.asarray(nodes,dtype=float)

  if centers is None:
    centers = nodes
  else:
    centers = np.asarray(centers,dtype=float)

  if order == 'max':
    order = rbf.poly.maximum_order(*nodes.shape)

  elif order is None:
    order = _default_poly_order(nodes.shape[0],nodes.shape[1])
    
  if diffs is not None:
    diffs = [tuple(d) for d in diffs]
    if len(coeffs) != len(diffs):
      raise ValueError('length of coeffs must equal length of diffs')

  elif diff is not None:
    diffs = [tuple(diff)]
    coeffs = [1.0]
    
  else:
    raise ValueError('must specify either diff or diffs')
    
  powers = rbf.poly.monomial_powers(order,nodes.shape[1])
  if powers.shape[0] > nodes.shape[0]:
    raise ValueError(
      'the number of monomials exceeds the number of RBFs for the '
      'stencil. Lower the polynomial order or ' 
      'increase the stencil size')
    
  # expand eps from scalar to array
  arr = np.empty(centers.shape[0])
  arr[:] = eps
  eps = arr
  
  # left hand side
  lhs = _arbf(nodes,centers,eps,powers,basis)
  # if diff is a DiffExpression instance
  rhs = np.zeros(centers.shape[0] + powers.shape[0])
  for c,d in zip(coeffs,diffs):
    rhs += c*_drbf(x,centers,eps,powers,d,basis)

  try:
    weights = np.linalg.solve(lhs,rhs)[:nodes.shape[0]]
  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute RBF-FD weight for point %s. Make sure that the '
       'stencil meets the conditions for non-singularity. This error '
       'may also be due to numerically flat basis functions' % x)

  return weights 


def poly_diff_weights(x,nodes,diff=None,diffs=None,coeffs=None):
  ''' 
  returns the traditional 1-D finite difference weights derived 
  from polynomial expansion. The input must have one spatial dimension
  
  Parameters
  ----------
    x : (1,) array

    nodes : (N,1) array

    diff : (1,) int array 

    diffs : (N,1) int array

    coeffs : (N,) array
        
  '''
  x = np.asarray(x)
  nodes = np.asarray(nodes)

  if len(x.shape) != 1:
    raise ValueError('x must be a 1-D array')

  if len(nodes.shape) != 2:
    raise ValueError('nodes must be a 2-D array')
    
  if x.shape[0] != 1:
    raise ValueError('x must have one spatial dimension to compute a poly-FD weight')

  if nodes.shape[1] != 1:
    raise ValueError('nodes must have one spatial dimension to compute a poly-FD weight')
    
  if diffs is not None:
    diffs = [tuple(d) for d in diffs]
    if len(coeffs) != len(diffs):
      raise ValueError('length of coeffs must equal length of diffs')

  elif diff is not None:
    diffs = [tuple(diff)]
    coeffs = [1.0]
    
  else:
    raise ValueError('must specify either diff or diffs')

  order = rbf.poly.maximum_order(*nodes.shape)
  powers = rbf.poly.monomial_powers(order,1)
  lhs = rbf.poly.mvmonos(nodes,powers,diff=(0,),check_input=False).T
  rhs = np.zeros(nodes.shape[0])
  for c,d in zip(coeffs,diffs):
    rhs += c*rbf.poly.mvmonos(x[None,:],powers,diff=d,check_input=False)[0,:]
  
  try:
    weights = np.linalg.solve(lhs,rhs)

  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute poly-FD weight for point %s. Make sure that '
       'the stencil meets the conditions for non-singularity. ' % x)

  return weights 


def _default_stencil_size(nodes,dim,diff=None,diffs=None):
  max_size = nodes
  if diff is not None:
    max_order = sum(diff)

  elif diffs is not None:
    max_order = max(sum(d) for d in diffs)

  else:
    raise ValueError('diff or diffs must be specified')
    
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
    

def diff_matrix(x,diff=None,diffs=None,coeffs=None,
                basis=rbf.basis.phs3,order=None,
                N=None,vert=None,smp=None):
  ''' 
  convenience function for creating a stencil network and then making 
  a differentiation matrix using RBF-FD weights.   
  
  If x is 1-D then stencil_network_1d is used. stencil_network_1d is 
  faster and it provides better connectivity than stencil_network
  
  ''' 
  x = np.asarray(x)
  
  if N is None:
    N = _default_stencil_size(x.shape[0],x.shape[1],diff=diff,diffs=diffs)
    
  if order is None:
    order = _default_poly_order(N,x.shape[1])
    
  sn = rbf.stencil.stencil_network(x,N=N,vert=vert,smp=smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)

  # determine whether to use the polynomial weight function
  for i,si in enumerate(sn):
    data[i,:] = diff_weights(x[i],x[si],diff=diff,
                             diffs=diffs,coeffs=coeffs,
                             basis=basis,order=order)

  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],x.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)

  return L

def poly_diff_matrix(x,diff=None,diffs=None,coeffs=None,
                     N=None,vert=None,smp=None):
  ''' 

  convenience function for creating a stencil network and then making 
  a differentiation matrix using the traditional FD weights.  The 
  stencil is determined by adjacency rather than nearest neighbors, 
  which results in better network connectivity

  '''
  x = np.asarray(x) 

  if N is None:
    N = _default_stencil_size(x.shape[0],x.shape[1],diff=diff,diffs=diffs)
    
  sn = rbf.stencil.stencil_network_1d(x,N=N,vert=vert,smp=smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
  for i,si in enumerate(sn):
    data[i,:] = poly_diff_weights(x[i],x[si],diff=diff,
                                  diffs=diffs,coeffs=coeffs)
    
  rows = np.repeat(range(data.shape[0]),data.shape[1])
  cols = sn.ravel()
  data = data.ravel()
  size = x.shape[0],x.shape[0]
  L = scipy.sparse.csr_matrix((data,(rows,cols)),size)

  return L

def grid_diff_matrices(Lx,Ly):
  ''' 
  DEPRICATED!!!
  
  Consider the array u with shape (Nx,Ny).  Lx is a (Nx,Nx) 
  differentiation matrix which acts along the rows of u. Ly is a 
  (Ny,Ny) differentiation matrix which acts along the columns of u. 
  This function returns two matrix where one performs the action of Lx 
  on a flattened u and the other performs the action of Ly on a 
  flattened u.

  Parameters
  ----------
    Lx: (Rx,Cx) sparse matrix
    Ly: (Ry,Cy) sparse matrix

  Returns
  -------
    Lxy1: (Rx*Cy,Cx*Cy) matrix which performs the action of Lx
    Lxy2: (Ry*Cx,Cx*Cy) matrix which performs the action of Ly

  '''
  if not (scipy.sparse.isspmatrix(Lx) & scipy.sparse.isspmatrix(Ly)):
    raise TypeError('smoothing matrices must be sparse')

  Rx,Cx = Lx.shape
  Ry,Cy = Ly.shape

  # this format allows me to extract the rows, columns, and values of 
  # each entry
  Lx = Lx.tocoo()
  Ly = Ly.tocoo()
  rx,cx,vx = Lx.row,Lx.col,Lx.data
  ry,cy,vy = Ly.row,Ly.col,Ly.data

  # create row and column indices for the expanded matrix
  rx *= Cy
  rx = np.repeat(rx[:,None],Cy,axis=1)
  rx += np.arange(Cy)
  rx = rx.ravel()

  cx *= Cy
  cx = np.repeat(cx[:,None],Cy,axis=1)
  cx += np.arange(Cy)
  cx = cx.ravel()

  vx = np.repeat(vx[:,None],Cy,axis=1)
  vx = vx.ravel()

  Lxy1 = scipy.sparse.csr_matrix((vx,(rx,cx)),(Rx*Cy,Cx*Cy))

  ry = np.repeat(ry[:,None],Cx,axis=1)
  ry += np.arange(Cx)*Ry
  ry = ry.ravel()

  cy = np.repeat(cy[:,None],Cx,axis=1)
  cy += np.arange(Cx)*Cy
  cy = cy.ravel()

  vy = np.repeat(vy[:,None],Cx,axis=1)
  vy = vy.ravel()

  Lxy2 = scipy.sparse.csr_matrix((vy,(ry,cy)),(Ry*Cx,Cx*Cy))

  return Lxy1,Lxy2

                 

