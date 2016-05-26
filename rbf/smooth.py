#!/usr/bin/env python
''' 
This module is for solving smoothing problems of the form

  | I |            | u_obs |
  | L | u_smooth = | 0     |

'''
import numpy as np
import rbf.interpolant
import rbf.stencil
import rbf.weights
import scipy.sparse
import modest.solvers
import modest.cv
import logging
logger = logging.getLogger(__name__)


def laplacian_diff_op(dim):
   return [(1.0,tuple(i)) for i in 2*np.eye(dim,dtype=int)]


def smoothing_matrix(x,connectivity=None,stencil_size=10,
                     diff=None,basis=rbf.basis.phs3,order=1,
                     vert=None,smp=None):
  ''' 
  Description
  -----------
    Returns the smoothing matrix L

  Parameters
  ----------
    x: (N,D) array of observation points

    connectivity (default=None): used to determine the stencil size.
      If specified, this overrides any argument for stencil_size

    stencil_size (default=10): number of nodes in each finite 
      difference stencil. 

    diff (default=None): smoothing differential operator. If not 
      specified then the D-dimensional Laplacian is used

    basis (default=phs3): radial basis function used to compute finite
      difference weights

    order (default=1): polynomial order used to compute finite 
      difference weights

    vert (default=None): boundary vertices. The smoothness is not 
      imposed across this boundary
    
    smp (default=None): connectivity of the boundary vertices
    

  Returns
  -------
    L: CSR sparse matrix

  '''
  # normalize the observation points for numerical stability
  norm = rbf.interpolant.DomainNormalizer(x)
  x = norm(x)
  if vert is not None:
    vert = norm(vert)

  N,D = x.shape

  # form stencil
  s = rbf.stencil.stencils(x,C=connectivity,N=stencil_size,
                           vert=vert,smp=smp)
  # get stencil size
  Ns = s.shape[1]

  if diff is None:
    diff = laplacian_diff_op(D)
  
  # values that will be put into the sparse matrix
  data = np.zeros((N,Ns),dtype=float)
  for i,si in enumerate(s):
    data[i,:] = rbf.weights.rbf_weight(x[i],x[si],diff=diff,
                                       basis=basis,order=order)
  rows = np.repeat(range(N),Ns)
  cols = s.ravel()
  data = data.ravel()
  L = scipy.sparse.csr_matrix((data,(rows,cols)))

  return L  


def grid_smoothing_matrices(Lx,Ly):
  ''' 
  Description
  -----------
    Consider the array u with shape (Nx,Ny).  Lx is a (Nx,Nx) 
    smoothing matrix which acts along the rows of u. Ly is a (Ny,Ny) 
    smoothing matrix which acts along the columns of u. This function 
    returns a matrix, Lxy, which imposes both the smoothing 
    constraints in Lx and Ly on a flattened u.

  Parameters
  ----------
    Lx: (Rx,Cx) sparse matrix
    Ly: (Ry,Cy) sparse matrix

  Returns
  -------
    Lxy: (Rx*Cy+Ry*Cx,Cx*Cy) CSR sparse matrix. The top half imposes the 
      Lx constraints and the bottom half imposes the Ly constraints
      
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

  #Lxy = scipy.sparse.vstack((Lxy1,Lxy2))

  return Lxy1,Lxy2
  

def smooth_data(data,L,damping=None,sigma=None,**kwargs):
  ''' 
  Description
  -----------
    solves for the smoothed data. By default a direct solver is used.  
    If the system is too large then an iterative solver should be used 
    by setting 'direct' to False.  If PETSc is available then the PETSc 
    iterative solvers will be used and any kwargs will be passed to the 
    function 'petsc_solve'. If PETSc is not available then scipy's lgmres
    solver will be used and kwargs will be passed to that function

  Parameters
  ----------
    L: CSR sparse smoothing matrix 

    data: vector of data to be smoothed

    damping: penalty parameter

    dsolve (default=True): whether to use a direct solver
  
    use_petsc (default=True): controls whether to use PETSc for 
      iterative solvers. this has no effect if dsolve is True

  Returns
  -------
    soln: vector of smoothed data
  '''
  is_sparse = scipy.sparse.isspmatrix(L)
  if not is_sparse:
    L = np.ararray(L)

  data = np.asarray(data)

  N = data.shape[0]

  if sigma is None:
    sigma = np.ones(N)

  sigma = np.asarray(sigma)

  # build system matrix
  if is_sparse:
    I = scipy.sparse.diags(1.0/sigma,0)
  else:
    I = np.diag(1.0/sigma)
    
  data = data/sigma
  
  if damping is None:
    damping = modest.cv.optimal_damping_parameter(I,L,data)[0]

  if is_sparse:  
    soln = modest.solvers.sparse_reg_ds(I,damping*L,data)
  else:
    soln = modest.solvers.reg_ds(I,L,data)

  return soln
