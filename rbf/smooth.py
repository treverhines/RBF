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
import sys
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

try:
  import scikits.umfpack
  HAS_UMFPACK = True
except ImportError:
  print(
    'could not import umfpack. The default scipy direct solver will be '
    'used instead. umfpack is a module within scikit that can be '
    'installed by following the instructions at\n\n'
    'http://scikit-umfpack.github.io/scikit-umfpack/\n')
  HAS_UMFPACK = False

try:
  import petsc4py
  petsc4py.init()
  from petsc4py import PETSc
  HAS_PETSC = True
except ImportError:
  print(
    'could not import PETSc. The scipy sparse solvers will be used instead. '
    'PETSc can be installed by following the instructions at '
    'https://www.mcs.anl.gov/petsc. Interfacing with PETSc requires '
    'petsc4py which can be found at https://bitbucket.org/petsc/petsc4py. '
    'Installing the latest version of petsc4py can be done with the command\n\n'
    '  pip install https://bitbucket.org/petsc/petsc4py/get/master.tar.gz\n')
  HAS_PETSC = False  


CONVERGENCE_REASON = {
  1:'KSP_CONVERGED_RTOL_NORMAL',
  9:'KSP_CONVERGED_ATOL_NORMAL',
  2:'KSP_CONVERGED_RTOL',
  3:'KSP_CONVERGED_ATOL',
  4:'KSP_CONVERGED_ITS',
  5:'KSP_CONVERGED_CG_NEG_CURVE',
  6:'KSP_CONVERGED_CG_CONSTRAINED',
  7:'KSP_CONVERGED_STEP_LENGTH',
  8:'KSP_CONVERGED_HAPPY_BREAKDOWN',
  -2:'KSP_DIVERGED_NULL',
  -3:'KSP_DIVERGED_ITS',
  -4:'KSP_DIVERGED_DTOL',
  -5:'KSP_DIVERGED_BREAKDOWN',
  -6:'KSP_DIVERGED_BREAKDOWN_BICG',
  -7:'KSP_DIVERGED_NONSYMMETRIC',
  -8:'KSP_DIVERGED_INDEFINITE_PC',
  -9:'KSP_DIVERGED_NANORINF',
  -10:'KSP_DIVERGED_INDEFINITE_MAT',
  -11:'KSP_DIVERGED_PCSETUP_FAILED',
  0:'KSP_CONVERGED_ITERATING'}


def petsc_solve(G,d,solver='lgmres',pc='jacobi',rtol=1e-6,atol=1e-6,maxiter=10000):
  ''' 
  Description
  -----------
    Solves a linear system using PETSc

  Parameters
  ----------
    G: (N,N) CSR sparse matrix
    d: (N,) data vector

    solver (default='preonly'): solve the system with this PETSc 
      routine. See PETSc documentation for a complete list of options.  
      'preonly' means that the system is solved with just the 
      preconditioner. This is done when the preconditioner is 'lu', 
      which means that the system is directly solved with LU 
      factorization. If the system is too large to allow for a direct 
      solution then use an iterative solver such as 'lgmres' or 'gmres'

    pc (default='lu'): type of preconditioner. See PETSc documentation 
      for a complete list of options. 'jacobi' seems to work best for 
      iterative solvers. Use 'lu' if the solver is 'preonly'

    rtol: relative tolerance for iterative solvers
 
    atol: absolute tolerance for iterative solvers
  
    maxiter: maximum number of iterations

  '''
  # instantiate LHS
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data)) 

  # instantiate RHS
  d = PETSc.Vec().createWithArray(d)

  # create empty solution vector
  soln = np.zeros(G.shape[1])
  soln = PETSc.Vec().createWithArray(soln)

  # instantiate solver
  ksp = PETSc.KSP()
  ksp.create()
  ksp.setType(solver)
  ksp.getPC().setType(pc)
  ksp.setOperators(A)
  ksp.setTolerances(rtol=rtol,atol=atol,max_it=maxiter)

  # solve and get information
  ksp.solve(d,soln)
  #ksp.view()
  conv_number = ksp.getConvergedReason()
  conv_reason = CONVERGENCE_REASON[conv_number]
  if conv_number > 0:
    logger.info('KSP solver converged due to %s' % conv_reason)
  else:
    logger.warning('KSP solver diverged due to %s' % conv_reason)
    print('WARNING: KSP solver diverged due to %s' % conv_reason)
   
  return soln.getArray()


def scipy_solve(G,data,**kwargs):
  ''' 
  calls LGMRES and prints necessary info
  '''
  soln,info = scipy.sparse.linalg.lgmres(G,data,**kwargs)
  if info < 0:
    logger.warning('LGMRES exited with value %s' % info)

  elif info == 0:
    logger.info('LGMRES finished successfully')

  elif info > 0:
    logger.warning('LGMRES did not converge after %s iterations' % info)
    print('WARNING: LGMRES did not converge after %s iterations' % info)

  return soln


def chunkify(list,N):
    return [list[i::N] for i in range(N)]


def laplacian_diff_op(dim):
   return [(1.0,tuple(i)) for i in 2*np.eye(dim,dtype=int)]


def smoothing_matrix(x,connectivity=1,stencil_size=None,
                     diff=None,basis=rbf.basis.phs3,order=1,
                     vert=None,smp=None):
  ''' 
  Description
  -----------
    Returns the smoothing matrix L

  Parameters
  ----------
    x: (N,D) array of observation points

    connectivity (default=1): used to determine the stencil size
      unless stencil size is explicitly given as an argument

    stencil size (default=None): number of nodes in each finite 
      difference stencil. Overrides connectivity if specified

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
  cols = s.flatten()
  data = data.flatten()
  L = scipy.sparse.csr_matrix((data,(rows,cols)))

  return L  


def predictive_error(L,data,damping,fold=10,dsolve=True,use_petsc=HAS_PETSC,**kwargs):
  ''' 
  returns predictive error for cross validation
  '''
  if not scipy.sparse.isspmatrix_csr(L):
    raise TypeError('smoothing matrix must be CSR sparse')
  
  N = data.shape[0]
  K = L.shape[0]
  # make sure folds is smaller than the number of data points
  fold = min(fold,N)
  res = np.zeros(N)

  # G = (I + damping**2*LtL)
  # the LtL step is done here fore efficiency
  G = damping**2*L.T.dot(L)
  I = scipy.sparse.eye(N)
  d = np.copy(data)
  for rmidx in chunkify(range(N),fold):
    # data can effectively be removed by setting the appropriate 
    # elements of I and data to zero

    # set lhs
    I.data[0][rmidx] = 0.0 # make elements 'rmidx' of the main diag zero
    G += I # form (I + damping**2*LtL)
    # set rhs
    d[rmidx] = 0.0 

    # smoothed data
    if dsolve:
      soln = scipy.sparse.linalg.spsolve(G,d,**kwargs)

    else:  
      if HAS_PETSC & use_petsc:
        soln = petsc_solve(G,d,**kwargs) 
      else:
        soln = scipy_solve(G,d,**kwargs)

    # reset lhs
    G -= I 
    I.data[0][rmidx] = 1.0
    # reset rhs
    d[rmidx] = data[rmidx]

    res[rmidx] = soln[rmidx] - data[rmidx]

  return res.dot(res)/N


def smoothed_data(L,data,damping,dsolve=True,use_petsc=HAS_PETSC,**kwargs):
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
  if not scipy.sparse.isspmatrix_csr(L):
    raise TypeError('smoothing matrix must be CSR sparse')

  N = data.shape[0]
  I = scipy.sparse.eye(N)
  G = I + damping**2*L.T.dot(L)

  if dsolve:
    soln = scipy.sparse.linalg.spsolve(G,data,**kwargs)

  else:  
    if HAS_PETSC & use_petsc:
      soln = petsc_solve(G,data,**kwargs) 
    else:
      soln = scipy_solve(G,data,**kwargs)
  
  return soln


def optimal_damping(L,data,plot=False,fold=10,log_bounds=None,itr=100,**kwargs):
  ''' 
  returns the optimal penalty parameter for regularized least squares 
  using generalized cross validation
  
  Parameters
  ----------
    L: (K,M) smoothing matrix
    data: (N,) data vector
    plot: whether to plot the predictive error curve

  '''
  if log_bounds is None:
    log_bounds = (-6.0,6.0)

  alpha_range = 10**np.linspace(log_bounds[0],log_bounds[1],itr)
  # predictive error for all tested damping parameters
  err = np.array([predictive_error(L,data,a,fold=fold,**kwargs)
                  for a in alpha_range])
  optimal_alpha = alpha_range[np.argmin(err)]
  optimal_err = np.min(err)
  if plot:
    fig,ax = plt.subplots()
    ax.set_title('%s-fold cross validation curve' % fold)
    ax.set_ylabel('predictive error')
    ax.set_xlabel('penalty parameter')
    ax.loglog(alpha_range,err,'k-')
    ax.loglog(optimal_alpha,optimal_err,'ko',markersize=10)
    ax.grid()

  return optimal_alpha
 

