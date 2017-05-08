''' 
This module contains light wrappers for some of the LAPACK functions
exposed through *scipy.linalg.lapack*. These functions perform no data
checks and should be used when every microsecond counts.
'''
import numpy as np
from scipy.linalg.lapack import dpotri,dpotrf,dtrtrs,dgetrf,dgetrs 


def solve_lu(A,b,overwrite_a=False,overwrite_b=False):
  ''' 
  Solves the system of equations *Ax = b* using the LU routines
  *dgetrf* and *dgetrs*.
  
  Parameters
  ----------
  A : (N,N) float array
  b : (N,*) float array
  '''
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  lu,piv,info = dgetrf(A,overwrite_a=overwrite_a)
  # I am too lazy to look up the error codes
  if info != 0:
    raise np.linalg.LinAlgError(
      'LAPACK routine *dgetrf* exited with error code %s' % info)

  x,info = dgetrs(lu,piv,b,overwrite_b=overwrite_b)
  if info != 0:
    raise np.linalg.LinAlgError(
      'LAPACK routine *dgetrs* exited with error code %s' % info)

  return x


def solve_triangular(L,b,lower=True):
  ''' 
  Solve the triangular system of equations *Lx = b* using *dtrtrs*.

  Parameters
  ----------
  L : (N,N) float array
  b : (N,*) float array
  '''
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x,info = dtrtrs(L,b,lower=lower)
  if info < 0:
    raise ValueError(
      'The %s-th argument had an illegal value' % (-info))

  elif info > 0:
    raise np.linalg.LinAlgError(
      'The %s-th diagonal element of A is zero, indicating that '
      'the matrix is singular and the solutions X have not been '
      'computed.' % info)

  return x


def cholesky(A,lower=True):
  ''' 
  Computes the Cholesky decomposition of *A* using the routine
  *dpotrf*.

  Parameters
  ----------
  A : (N,N) float array
  lower : bool, optional
  '''
  if A.shape == (0,0):
    return np.zeros((0,0),dtype=float)

  L,info = dpotrf(A,lower=lower)
  if info > 0:
    raise np.linalg.LinAlgError(
      'The leading minor of order %s is not positive definite, and '
      'the factorization could not be completed. ' % info)

  elif info < 0:
    raise ValueError(
      'The %s-th argument has an illegal value.' % (-info))

  return L


def cholesky_inv(A):
  ''' 
  Returns the inverse of the positive definite matrix *A* using
  *_cholesky* and *dpotri*.

  Parameters
  ----------
  A : (N,N) float array
  '''
  if A.shape == (0,0):
    return np.zeros((0,0),dtype=float)

  L = cholesky(A,lower=True)
  # Linv is the lower triangular components of the inverse.
  Linv,info = dpotri(L,lower=True)
  if info < 0:
    raise ValueError(
      'The %s-th argument had an illegal value.' % (-info))

  elif info > 0:
    raise np.linalg.LinAlgError(
      'The (%s,%s) element of the factor U or L is zero, and the '
      'inverse could not be computed.' % (info,info))

  else:
    # the decomposition exited successfully
    # reflect Linv over the diagonal      
    Linv = Linv + Linv.T
    # the diagonals are twice as big as they should be
    diag = Linv.diagonal()
    diag.flags.writeable = True
    diag *= 0.5
    return Linv
