from __future__ import division
import numpy as np
import rbf.basis
import rbf.poly
import rbf.stencil
import scipy.sparse
import logging

logger = logging.getLogger(__name__)

def _apoly(nodes,order):
  ''' 
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


def _dpoly(x,order,diff):
  ''' 
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
  powers = rbf.poly.monomial_powers(order,x.shape[1])
  out = rbf.poly.mvmonos(x,powers,diff)[0,:]
  
  return out


def _arbf(nodes,centers,eps,order,basis):
  ''' 
  Returns the matrix:

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
  Ap = _apoly(centers,order)  
  A[Ns:,:Ns] = Ap
  A[:Ns,Ns:] = Ap.T
  return A


def _drbf(x,centers,eps,order,diff,basis): 
  ''' 
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
  d[Ns:] = _dpoly(x[0,:],order,diff)

  return d


def diff_weights(x,nodes,diff=None,
                 diffs=None,coeffs=None,centers=None,
                 basis=rbf.basis.phs3,order=1,
                 eps=1.0,diff_args=None):
  ''' 
  computes the weights used for a finite difference approximation at x

  The weights are computed using the RBF-FD method described in "A 
  Primer on Radial Basis Functions with Applications to the 
  Geosciences" by Bengt Fornberg and Natasha Flyer.  

  Parameters
  ----------
    x: (D,) position where the derivative is being approximated

    nodes: (N,D) nodes adjacent to x

    diff: (D,) tuple of derivative orders for each spatial dimension. 

    centers: (N,D) centers of each radial basis function. If not 
      specified, then the nodes will be used as centers. This is often 
      used when trying out exotic ways of imposing boundary conditions.
   
    basis: radial basis function to use. Select from those available 
      in rbf.basis
 
    order: order of added polynomial terms.  can be 'max' to use the 
      largest number of polynomials without creating a singular 
      matrix.  This may lead to lead to instabilities in derivative 
      approximations. 1 is generally a safe value   

    eps: shape parameter. This only makes a difference when using 
      RBFs that are not scale invariant, which you should not do. Any 
      of the odd PHS basis function are unaffected by the shape parameter. 
      However, if the problem is particularly poorly scaled then eps 
      may be a good way to scale the problem to something sensible.
    
    diffs: list of derivative tuples. if specified then it overrides 
      diff and coeffs must also be specified

    coeffs: list of coefficients for each derivative in diffs. does 
      nothing if diffs is not specified

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

  '''
  x = np.asarray(x,dtype=float)
  nodes = np.asarray(nodes,dtype=float)
  if centers is None:
    centers = nodes

  centers = np.asarray(centers,dtype=float)

  if order == 'max':
    order = rbf.poly.maximum_order(*nodes.shape)


  # number of polynomial terms that will be used
  Np = rbf.poly.monomial_count(order,x.shape[0])
  if Np > nodes.shape[0]:
    raise ValueError(
      'the number of monomials exceeds the number of RBFs for the '
      'stencil. Lower the polynomial order or ' 
      'increase the stencil size')

  # left hand side
  lhs = _arbf(nodes,centers,eps,order,basis)
  # if diff is a DiffExpression instance
  if diffs is not None:
    if len(diffs) != len(coeffs):
      raise ValueError(
        'length of coeffs must equal the length of diffs when diffs '
        'is specified')

    rhs = np.zeros(nodes.shape[0] + Np)
    for c,d in zip(coeffs,diffs):
      rhs += c*_drbf(x,centers,eps,order,d,basis)
  
  elif diff is not None:
    rhs = _drbf(x,centers,eps,order,diff,basis)

  else:
    raise ValueError('must specify either diff or diffs')

  try:
    weights = np.linalg.solve(lhs,rhs)[:nodes.shape[0]]
  except np.linalg.LinAlgError:
     raise np.linalg.LinAlgError(
       'cannot compute RBF-FD weight for point %s. Make sure that the '
       'stencil meets the conditions for non-singularity. This error '
       'may also be due to numerically flat basis functions' % x)

  return weights 


def diff_matrix(x,diff=None,diffs=None,coeffs=None,
                basis=rbf.basis.phs3,order=1,
                C=None,N=None,vert=None,smp=None):
  ''' 
  convenience function for creating a stencil network and then making 
  differentiation matrix. If x is 1-D then stencil_network_1d is used
  
  '''
  x = np.asarray(x)
  if x.shape[1] == 1:  
    sn = rbf.stencil.stencil_network_1d(x,C=C,N=N,vert=vert,smp=smp)
  else:
    sn = rbf.stencil.stencil_network(x,C=C,N=N,vert=vert,smp=smp)

  # values that will be put into the sparse matrix
  data = np.zeros(sn.shape,dtype=float)
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


def poly_weight(x,nodes,diff):
  ''' 
  DONT USE THIS FUNCTION

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
  A =  _apoly(nodes,order)
  d =  _dpoly(x,order,diff) 
  w = np.linalg.solve(A,d)
  return w 

