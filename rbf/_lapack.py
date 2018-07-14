''' 
This module contains light wrappers for some of the LAPACK functions
exposed through *scipy.linalg.lapack*. These functions perform no data
checks and should be used when every microsecond counts.
'''
import numpy as np
from scipy.linalg.lapack import (dpotrf, dpotrs, dtrtrs, dgetrf,
                                 dgetrs)

def solve(A,b,positive_definite=False):
  ''' 
  Solves the system of equations `Ax = b` using either the LU routines
  or Cholesky routines, depending on whether `positive_definite` is
  `True`.
  
  Parameters
  ----------
  A : (N,N) float array
  b : (N,*) float array
  positive_definite : bool

  Returns
  -------
  (N,*) float array
  '''
  if positive_definite:
    chol = cholesky(A,lower=True)         
    x = solve_cholesky(chol,b,lower=True)
  
  else:
    fac,piv = lu(A)
    x = solve_lu(fac,piv,b)
  
  return x  


def lu(A):
  '''
  Computes the LU factorization of `A` using the routine `dgetrf`

  Parameters
  ----------
  A : (N,N) float array

  Returns
  -------
  (N,N) float array
    LU factorization

  (N,) int array
    pivots
  '''
  # handle rank zero matrix
  if A.shape == (0,0):
    return (np.zeros((0,0),dtype=float), 
            np.zeros((0,),dtype=np.int32))
            
  # get the LU factorization  
  fac,piv,info = dgetrf(A)
  if info < 0:
    raise ValueError(
      'the %s-th argument had an illegal value' % -info)

  elif info > 0:
    raise np.linalg.LinAlgError(
      'U(%s,%s) is exactly zero. The factorization '
      'has been completed, but the factor U is exactly '
      'singular, and division by zero will occur if it is used '
      'to solve a system of equations. ' % (info,info))

  return fac,piv


def solve_lu(fac,piv,b):
  '''
  Solves the system of equations `Ax = b` given the LU factorization
  of `A`. Uses the `dgetrs` routine.

  Parameters
  ----------
  fac : (N,N) float array
  piv : (N,) int array
  b : (N,*) float array

  Returns
  -------
  (N,*) float array
  '''
  # handle the case of an array with zero-length for an axis.
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x,info = dgetrs(fac,piv,b)
  if info != 0:
    raise ValueError(
      'the %s-th argument had an illegal value' % -info)

  return x

def cholesky(A,lower=True):
  ''' 
  Computes the Cholesky decomposition of `A` using the routine
  `dpotrf`.

  Parameters
  ----------
  A : (N,N) float array
  lower : bool, optional

  Returns
  -------
  (N,N) float array
  '''
  # handle rank zero matrix
  if A.shape == (0,0):
    return np.zeros((0,0),dtype=float)

  L,info = dpotrf(A,lower=lower)
  if info > 0:
    raise np.linalg.LinAlgError(
      'The leading minor of order %s is not positive definite, and '
      'the factorization could not be completed. ' % info)

  elif info < 0:
    raise ValueError(
      'The %s-th argument has an illegal value.' % -info)

  return L


def solve_cholesky(L,b,lower=True):
  ''' 
  Solves the system of equations `Ax = b` given the Cholesky
  decomposition of `A`. Uses the routine `dpotrs`.

  Parameters
  ----------
  L : (N,N) float array
  b : (N,*) float array

  Returns
  -------
  (N,*) float array
  '''
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x,info = dpotrs(L,b,lower=lower)
  if info < 0:
    raise ValueError(
      'The %s-th argument has an illegal value.' % -info)
  
  return x
  

def solve_triangular(L,b,lower=True):
  ''' 
  Solve the triangular system of equations `Lx = b` using `dtrtrs`.

  Parameters
  ----------
  L : (N,N) float array
  b : (N,*) float array

  Returns
  -------
  (N,*) float array
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


