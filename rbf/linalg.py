'''
Module that defines solvers for matrices that are frequently encountered in RBF
applications. The classes in this module can take either scipy sparse matrices
or numpy arrays as input. The underlying algorithms are from
`scipy.linalg.lapack`, `scipy.sparse.linalg`, and `sksparse.cholmod`.
'''
import logging
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg.lapack import (dpotrf, dpotrs, dtrtrs, dgetrf,
                                 dgetrs)

from rbf.sputils import row_norms, divide_rows

LOGGER = logging.getLogger(__name__)

try:
  from sksparse import cholmod
  HAS_CHOLMOD = True

except ImportError:
  HAS_CHOLMOD = False
  CHOLMOD_MSG = (
    'Could not import CHOLMOD. Sparse matrices will be converted to dense for '
    'all Cholesky decompositions. To install CHOLMOD and its python wrapper, '
    'follow the instructions at https://scikit-sparse.readthedocs.io. '
    'Anaconda users can install CHOLMOD with the command `conda install -c '
    'conda-forge scikit-sparse`')
  LOGGER.debug(CHOLMOD_MSG)


## Wrappers for low level LAPACK functions. These are all a few microseconds
## faster than their corresponding functions in scipy.linalg
###############################################################################
def _lu(A):
  '''
  Computes the LU factorization of `A` using the routine `dgetrf`

  Parameters
  ----------
  A : (n, n) float array

  Returns
  -------
  (n, n) float array
    LU factorization

  (n,) int array
    pivots

  '''
  # handle rank zero matrix
  if A.shape == (0, 0):
    return (np.zeros((0, 0), dtype=float),
            np.zeros((0,), dtype=np.int32))

  # get the LU factorization
  fac, piv, info = dgetrf(A)
  if info < 0:
    raise ValueError('the %s-th argument had an illegal value' % -info)

  elif info > 0:
    raise np.linalg.LinAlgError(
      'U(%s, %s) is exactly zero. The factorization has been completed, but '
      'the factor U is exactly singular, and division by zero will occur if '
      'it is used to solve a system of equations. ' % (info, info))

  return fac, piv


def _solve_lu(fac, piv, b):
  '''
  Solves the system of equations `Ax = b` given the LU factorization of `A`.
  Uses the `dgetrs` routine.

  Parameters
  ----------
  fac : (n, n) float array

  piv : (n,) int array

  b : (n, ...) float array

  Returns
  -------
  (n, ...) float array

  '''
  # handle the case of an array with zero-length for an axis.
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x, info = dgetrs(fac, piv, b)
  if info != 0:
    raise ValueError('the %s-th argument had an illegal value' % -info)

  return x


def _cholesky(A, lower=True):
  '''
  Computes the Cholesky decomposition of `A` using the routine `dpotrf`.

  Parameters
  ----------
  A : (n, n) float array

  lower : bool, optional

  Returns
  -------
  (n, n) float array

  '''
  # handle rank zero matrix
  if A.shape == (0, 0):
    return np.zeros((0, 0), dtype=float)

  L, info = dpotrf(A, lower=lower)
  if info > 0:
    raise np.linalg.LinAlgError(
      'The leading minor of order %s is not positive definite, and the '
      'factorization could not be completed. ' % info)

  elif info < 0:
    raise ValueError('The %s-th argument has an illegal value.' % -info)

  return L


def _solve_cholesky(L, b, lower=True):
  '''
  Solves the system of equations `Ax = b` given the Cholesky decomposition of
  `A`. Uses the routine `dpotrs`.

  Parameters
  ----------
  L : (n, n) float array

  b : (n, ...) float array

  Returns
  -------
  (n, ...) float array

  '''
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x, info = dpotrs(L, b, lower=lower)
  if info < 0:
    raise ValueError('The %s-th argument has an illegal value.' % -info)

  return x


def _solve_triangular(L, b, lower=True):
  '''
  Solve the triangular system of equations `Lx = b` using `dtrtrs`.

  Parameters
  ----------
  L : (n, n) float array

  b : (n, ...) float array

  Returns
  -------
  (n, ...) float array

  '''
  if any(i == 0 for i in b.shape):
    return np.zeros(b.shape)

  x, info = dtrtrs(L, b, lower=lower)
  if info < 0:
    raise ValueError('The %s-th argument had an illegal value' % (-info))

  elif info > 0:
    raise np.linalg.LinAlgError(
      'The %s-th diagonal element of A is zero, indicating that the matrix is '
      'singular and the solutions X have not been computed.' % info)

  return x


#####################################################################
def as_sparse_or_array(A, dtype=None, copy=False):
  '''
  If `A` is a scipy sparse matrix then return it as a csc matrix. Otherwise,
  return it as an array.
  '''
  if sp.issparse(A):
    # This does not make a copy if A is csc, has the same dtype and copy is
    # false.
    A = sp.csc_matrix(A, dtype=dtype, copy=copy)

  else:
    A = np.array(A, dtype=dtype, copy=copy)

  return A


def as_array(A, dtype=None, copy=False):
  '''
  Return `A` as an array if it is not already. This properly handles when `A`
  is sparse.
  '''
  if sp.issparse(A):
    A = A.toarray()

  A = np.array(A, dtype=dtype, copy=copy)
  return A


class _SparseSolver(object):
  '''
  computes the LU factorization of the sparse matrix `A` with SuperLU.
  '''
  def __init__(self, A):
    LOGGER.debug(
      'computing the LU decomposition of a %s by %s sparse matrix with %s '
      'nonzeros ' % (A.shape + (A.nnz,)))
    self.factor = spla.splu(A)

  def solve(self, b):
    '''
    solves `Ax = b` for `x`
    '''
    return self.factor.solve(b)


class _DenseSolver(object):
  '''
  computes the LU factorization of the dense matrix `A`.
  '''
  def __init__(self, A):
    fac, piv = _lu(A)
    self.fac = fac
    self.piv = piv

  def solve(self, b):
    '''
    solves `Ax = b` for `x`
    '''
    return _solve_lu(self.fac, self.piv, b)


class Solver(object):
  '''
  Computes an LU factorization of `A` and provides a method to solve `Ax = b`
  for `x`. `A` can be a scipy sparse matrix or a numpy array.

  Parameters
  ----------
  A : (n, n) array or scipy sparse matrix

  build_inverse : bool, optional
    Whether to construct the inverse of `A`, as opposed to just factoring it

  '''
  def __init__(self, A, build_inverse=False):
    A = as_sparse_or_array(A, dtype=float)
    if sp.issparse(A):
      self._solver = _SparseSolver(A)
    else:
      self._solver = _DenseSolver(A)

    if build_inverse:
      self._inverse = self._solver.solve(np.eye(A.shape[0]))
    else:
      self._inverse = None
    
  def solve(self, b):
    '''
    solves `Ax = b` for `x`

    Parameters
    ----------
    b : (n, ...) array or sparse matrix

    Returns
    -------
    (n, ...) array

    '''
    b = as_array(b, dtype=float)
    if self._inverse is not None:
      return self._inverse.dot(b)
    else:  
      return self._solver.solve(b)


class _SparsePosDefSolver(object):
  '''
  Factors the sparse positive definite matrix `A` as `LL^T = A`. Note that `L`
  is NOT necessarily the lower triangular matrix from a Cholesky decomposition.
  Instead, it is structured to be maximally sparse. This class requires
  CHOLMOD.
  '''
  def __init__(self, A):
    LOGGER.debug(
      'computing the Cholesky decomposition of a %s by %s sparse matrix with '
      '%s nonzeros ' % (A.shape + (A.nnz,)))
    self.factor = cholmod.cholesky(
      A,
      use_long=False,
      ordering_method='default')
    # store the squared diagonal components of the cholesky factorization
    self.d = self.factor.D()
    # store the permutation array, which permutes `A` such that its cholesky
    # factorization is maximally sparse
    self.p = self.factor.P()

  def solve(self, b):
    '''
    solves `Ax = b` for `x`
    '''
    return self.factor.solve_A(b)

  def solve_L(self, b):
    '''
    Solves `Lx = b` for `x`
    '''
    if b.ndim == 1:
      s_inv = 1.0/np.sqrt(self.d)

    elif b.ndim == 2:
      # expand for broadcasting
      s_inv = 1.0/np.sqrt(self.d)[:, None]

    else:
      raise ValueError('`b` must be a one or two dimensional array')

    out = s_inv*self.factor.solve_L(b[self.p])
    return out

  def L(self):
    '''Return the factorization `L`'''
    L = self.factor.L()
    p_inv = np.argsort(self.p)
    out = L[p_inv]
    return out

  def log_det(self):
    '''Returns the log determinant of `A`'''
    out = np.sum(np.log(self.d))
    return out


class _DensePosDefSolver(object):
  '''
  Computes to Cholesky factorization of the dense positive definite matrix `A`.
  This uses low level LAPACK functions
  '''
  def __init__(self, A):
    self.chol = _cholesky(A, lower=True)

  def solve(self, b):
    '''
    Solves the equation `Ax = b` for `x`
    '''
    return _solve_cholesky(self.chol, b, lower=True)

  def solve_L(self, b):
    '''
    Solves the equation `Lx = b` for `x`, where `L` is the Cholesky
    decomposition.
    '''
    return _solve_triangular(self.chol, b, lower=True)

  def L(self):
    '''Returns the Cholesky decomposition of `A`'''
    return self.chol

  def log_det(self):
    '''Returns the log determinant of `A`'''
    out = 2*np.sum(np.log(np.diag(self.chol)))
    return out


class PosDefSolver(object):
  '''
  Factors the positive definite matrix `A` as `LL^T = A` and provides an
  efficient method for solving `Ax = b` for `x`. Additionally provides a method
  to solve `Lx = b`, get the log determinant of `A`, and get `L`. `A` can be a
  scipy sparse matrix or a numpy array.

  Parameters
  ----------
  A : (n, n) array or scipy sparse matrix
    Positive definite matrix

  build_inverse : bool, optional
    Whether to construct the inverse of `A`, as opposed to just factoring it

  '''
  def __init__(self, A, build_inverse=False):
    A = as_sparse_or_array(A, dtype=float)
    if sp.issparse(A) & (not HAS_CHOLMOD):
      warnings.warn(CHOLMOD_MSG)
      A = A.toarray()

    if sp.issparse(A):
      self._solver = _SparsePosDefSolver(A)
    else:
      self._solver = _DensePosDefSolver(A)

    if build_inverse:
      self._inverse = self._solver.solve(np.eye(A.shape[0]))
    else:
      self._inverse = None

  def solve(self, b):
    '''
    solves `Ax = b` for `x`

    Parameters
    ----------
    b : (n, ...) array or sparse matrix

    Returns
    -------
    (n, ...) array

    '''
    b = as_array(b, dtype=float)
    if self._inverse is not None:
      return self._inverse.dot(b)
    else:  
      return self._solver.solve(b)

  def solve_L(self, b):
    '''
    solves `Lx = b` for `x`

    Parameters
    ----------
    b : (n, ...) array or sparse matrix

    Returns
    -------
    (n, ...) array

    '''
    b = as_array(b, dtype=float)
    return self._solver.solve_L(b)

  def L(self):
    '''
    Returns the factorization `L`

    Returns
    -------
    (n, n) array or sparse matrix

    '''
    return self._solver.L()

  def log_det(self):
    '''
    Returns the log determinant of `A`

    Returns
    -------
    float

    '''
    return self._solver.log_det()


def is_positive_definite(A):
  '''
  Tests if `A` is positive definite. This is done by testing whether the
  Cholesky decomposition finishes successfully. `A` can be a scipy sparse
  matrix or a numpy array.
  '''
  try:
    PosDefSolver(A).L()

  except (np.linalg.LinAlgError, cholmod.CholmodNotPositiveDefiniteError):
    return False

  return True


class PartitionedSolver(object):
  '''
  Solves the system of equations

  .. math::
    \\left[
    \\begin{array}{cc}
      A   & B \\\\
      B^T & 0 \\\\
    \\end{array}
    \\right]
    \\left[
    \\begin{array}{c}
      x \\\\
      y \\\\
    \\end{array}
    \\right]
    =
    \\left[
    \\begin{array}{c}
      a \\\\
      b \\\\
    \\end{array}
    \\right]

  for `x` and `y`. This class builds the system and then factors it with an LU
  decomposition. As opposed to `PartitionedPosDefSolver`, `A` is not assumed to
  be positive definite. `A` can be a scipy sparse matrix or a numpy array. `B`
  can also be a scipy sparse matrix or a numpy array but it will be converted
  to a numpy array.

  Parameters
  ----------
  A : (n, n) array or sparse matrix

  B : (n, p) array or sparse matrix

  build_inverse : bool, optional
    Whether to construct the inverse of the block matrix, as opposed to just
    factoring it

  '''
  def __init__(self, A, B, build_inverse=False):
    # make sure A is either a csc sparse matrix or a float array
    A = as_sparse_or_array(A, dtype=float)
    # ensure B is dense
    B = as_array(B, dtype=float)
    n, p = B.shape
    if n < p:
      raise np.linalg.LinAlgError(
        'There are fewer rows than columns in `B`. This makes the block '
        'matrix singular, and its inverse cannot be computed.')

    # concatenate the A and B matrices
    if sp.issparse(A):
        Z = sp.csc_matrix((p, p), dtype=float)
        C = sp.vstack((sp.hstack((A, B)), sp.hstack((B.T, Z))))
    else:
        Z = np.zeros((p, p), dtype=float)
        C = np.vstack((np.hstack((A, B)), np.hstack((B.T, Z))))

    self._solver = Solver(C, build_inverse=build_inverse)
    self.n = n

  def solve(self, a, b):
    '''
    Solves for `x` and `y` given `a` and `b`.

    Parameters
    ----------
    a : (n, ...) array or sparse matrix

    b : (p, ...) array or sparse matrix

    Returns
    -------
    (n, ...) array

    (p, ...) array

    '''
    a = as_array(a, dtype=float)
    b = as_array(b, dtype=float)
    c = np.concatenate((a, b), axis=0)
    xy = self._solver.solve(c)
    x, y = xy[:self.n], xy[self.n:]
    return x, y


class PartitionedPosDefSolver(object):
  '''
  Solves the system of equations

  .. math::
    \\left[
    \\begin{array}{cc}
      A   & B \\\\
      B^T & 0 \\\\
    \\end{array}
    \\right]
    \\left[
    \\begin{array}{c}
      x \\\\
      y \\\\
    \\end{array}
    \\right]
    =
    \\left[
    \\begin{array}{c}
      a \\\\
      b \\\\
    \\end{array}
    \\right]

  for `x` and `y`, where `A` is a positive definite matrix. Rather than naively
  building and solving the system, this class partitions the inverse as

  .. math::
    \\left[
    \\begin{array}{cc}
      C   & D \\\\
      D^T & E \\\\
    \\end{array}
    \\right]

  where

  .. math::
    C = A^{-1} - (A^{-1} B) (B^T A^{-1} B)^{-1} (A^{-1} B)^T

  .. math::
    D = (A^{-1} B) (B^T A^{-1} B)^{-1}

  .. math::
    E = - (B^T A^{-1} B)^{-1}

  The inverse of `A` is not computed, but instead its action is performed by
  solving the Cholesky decomposition of `A`. `A` can be a scipy sparse matrix
  or a numpy array. `B` can also be either a scipy sparse matrix or a numpy
  array but it will be converted to a numpy array.

  Parameters
  ----------
  A : (n, n) array or sparse matrix

  B : (n, p) array or sparse matrix

  build_inverse : bool, optional
    Whether to construct the inverse matrices

  Note
  ----
  This class stores the factorization of `A`, which may be sparse, the dense
  matrix `A^-1 B`, and the dense factorization of `B^T A^-1 B`. If the number
  of columns in `B` is large then this may take up too much memory.

  '''
  def __init__(self, A, B, build_inverse=False):
    # make sure A is either a csc sparse matrix or a float array
    A = as_sparse_or_array(A, dtype=float)
    # convert B to dense if it is sparse
    B = as_array(B, dtype=float)
    n, p = B.shape
    if n < p:
      raise np.linalg.LinAlgError(
        'There are fewer rows than columns in `B`. This makes the block '
        'matrix singular, and its inverse cannot be computed.')

    self.n = n
    self._A_solver = PosDefSolver(A, build_inverse=build_inverse)
    self._AiB = self._A_solver.solve(B)
    self._BtAiB_solver = PosDefSolver(B.T.dot(self._AiB), 
                                      build_inverse=build_inverse)

    if build_inverse:
      E = -self._BtAiB_solver._inverse
      D = self._AiB.dot(self._BtAiB_solver._inverse)
      C = self._A_solver._inverse - D.dot(self._AiB.T)
      self._inverse = np.vstack((np.hstack((C, D)), np.hstack((D.T, E))))
    else:
      self._inverse = None

  def solve(self, a, b):
    '''
    Solves for `x` and `y` given `a` and `b`.

    Parameters
    ----------
    a : (n, ...) array or sparse matrix

    b : (p, ...) array or sparse matrix

    Returns
    -------
    (n, ...) array

    (p, ...) array

    '''
    a = as_array(a, dtype=float)
    b = as_array(b, dtype=float)
    if self._inverse is not None:
      c = np.concatenate((a, b), axis=0)
      xy = self._inverse.dot(c)
      x, y = xy[:self.n], xy[self.n:]
      return x, y
      
    else:
      Eb  = -self._BtAiB_solver.solve(b)
      Db  = -self._AiB.dot(Eb)
      Dta = self._BtAiB_solver.solve(self._AiB.T.dot(a))
      Ca  = self._A_solver.solve(a) - self._AiB.dot(Dta)
      x = Ca  + Db
      y = Dta + Eb
      return x, y


class GMRESSolver(object):
  '''
  Solves the system of equations `Ax = b` for `x` iteratively with GMRES and an
  incomplete LU decomposition.

  Parameters
  ----------
  A : (n, n) CSC sparse matrix

  drop_tol : float, optional
    Passed to `scipy.sparse.linalg.spilu`. This controls the sparsity of the
    ILU decomposition used for the preconditioner. It should be between 0 and
    1. smaller values make the decomposition denser but better approximates the
    LU decomposition. If the value is too large then you may get a "Factor is
    exactly singular" error.

  fill_factor : float, optional
    Passed to `scipy.sparse.linalg.spilu`. I believe this controls the memory
    allocated for the ILU decomposition. If this value is too small then memory
    will be allocated dynamically for the decomposition. If this is too large
    then you may get a memory error.

  normalize_inplace : bool
    If True and `A` is a csc matrix, then `A` is normalized in place.

  '''
  def __init__(self,
               A,
               drop_tol=0.005,
               fill_factor=2.0,
               normalize_inplace=False):
    # the spilu and gmres functions are most efficient with csc sparse. If the
    # matrix is already csc then this will do nothing
    A = sp.csc_matrix(A)
    n = row_norms(A)
    if normalize_inplace:
      divide_rows(A, n, inplace=True)
    else:
      A = divide_rows(A, n, inplace=False).tocsc()

    LOGGER.debug(
      'computing the ILU decomposition of a %s by %s sparse matrix with %s '
      'nonzeros ' % (A.shape + (A.nnz,)))
    ilu = spla.spilu(
      A,
      drop_rule='basic',
      drop_tol=drop_tol,
      fill_factor=fill_factor)
    LOGGER.debug('done')
    M = spla.LinearOperator(A.shape, ilu.solve)
    self.A = A
    self.M = M
    self.n = n

  def solve(self, b, tol=1.0e-10):
    '''
    Solve `Ax = b` for `x`

    Parameters
    ----------
    b : (n,) array

    tol : float, optional

    Returns
    -------
    (n,) array

    '''
    # solve the system using GMRES and define the callback function to
    # print info for each iteration
    def callback(res, _itr=[0]):
      l2 = np.linalg.norm(res)
      LOGGER.debug('GMRES error on iteration %s: %s' % (_itr[0], l2))
      _itr[0] += 1

    LOGGER.debug('solving the system with GMRES')
    x, info = spla.gmres(
      self.A,
      b/self.n,
      tol=tol,
      M=self.M,
      callback=callback)
    LOGGER.debug('finished GMRES with info %s' % info)
    return x
