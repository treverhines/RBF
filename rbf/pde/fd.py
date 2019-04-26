''' 
This module provides functions for generating RBF-FD weights
'''
from __future__ import division
import logging

import numpy as np
import scipy.sparse as sp

from rbf.basis import phs3, get_rbf
from rbf.poly import count, powers, mvmonos
from rbf.utils import assert_shape, Memoize, KDTree
from rbf.linalg import PartitionedSolver, as_array

logger = logging.getLogger(__name__)


def _reshape_diffs(diffs):
  ''' 
  turns diffs into a 2D array
  '''
  if diffs.ndim > 2:
    raise ValueError('`diffs` can only be a 1 or 2 dimensional array')

  D = diffs.shape[-1]
  out = diffs.reshape((-1, D))
  return out
  

def _default_poly_order(diffs):
  ''' 
  This sets the polynomial order equal to the largest derivative 
  order. So a constant and linear term (order=1) will be used when 
  approximating a first derivative. This is the smallest polynomial 
  order needed to overcome PHS stagnation error
  '''
  P = max(sum(d) for d in diffs)
  return P


@Memoize
def _max_poly_order(size, dim):
  ''' 
  Returns the maximum polynomial order allowed for the given stencil 
  size and number of dimensions
  '''
  order = -1
  while (count(order+1, dim) <= size):
    order += 1

  return order


def weights(x, s, diffs, coeffs=None,
            phi=phs3, order=None,
            eps=1.0):
  ''' 
  Returns the weights which map a functions values at `s` to an
  approximation of that functions derivative at `x`. The weights are
  computed using the RBF-FD method described in [1]. In this function
  `x` is a single point in D-dimensional space. Use `weight_matrix` to
  compute the weights for multiple point.

  Parameters
  ----------
  x : (D,) array
    Target point. The weights will approximate the derivative at this
    point.

  s : (N, D) array
    Stencil points. The derivative will be approximated with a
    weighted sum of values at this point.

  diffs : (D,) int array or (K, D) int array 
    Derivative orders for each spatial dimension. For example `[2, 0]`
    indicates that the weights should approximate the second
    derivative with respect to the first spatial dimension in
    two-dimensional space.  diffs can also be a (K, D) array, where
    each (D,) sub-array is a term in a differential operator. For
    example the two-dimensional Laplacian can be represented as
    `[[2, 0], [0, 2]]`.

  coeffs : (K,) array, optional 
    Coefficients for each term in the differential operator specified
    with `diffs`.  Defaults to an array of ones. If `diffs` was
    specified as a (D,) array then `coeffs` should be a length 1
    array.

  phi : rbf.basis.RBF instance or str, optional
    Type of RBF. Select from those available in `rbf.basis` or create
    your own.
 
  order : int, optional
    Order of the added polynomial. This defaults to the highest
    derivative order. For example, if `diffs` is `[[2, 0], [0, 1]]`,
    then order is set to 2.

  eps : float or (N,) array, optional
    Shape parameter for each RBF, which have centers `s`. This only 
    makes a difference when using RBFs that are not scale invariant. 
    All the predefined RBFs except for the odd order polyharmonic 
    splines are not scale invariant.

  Returns
  -------
  out : (N,) array
    RBF-FD weights
    
  Examples
  --------
  Calculate the weights for a one-dimensional second order derivative.

  >>> x = np.array([1.0]) 
  >>> s = np.array([[0.0], [1.0], [2.0]]) 
  >>> diff = (2,) 
  >>> weights(x, s, diff)
  array([ 1., -2., 1.])
    
  Calculate the weights for estimating an x derivative from three
  points in a two-dimensional plane

  >>> x = np.array([0.25, 0.25])
  >>> s = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]])
  >>> diff = (1, 0)
  >>> weights(x, s, diff)
  array([ -1., 1., 0.])
    
  Notes
  -----
  This function may become unstable with high order polynomials (i.e.,
  `order` is high). This can be somewhat remedied by shifting the
  coordinate system so that x is zero

  References
  ----------
  [1] Fornberg, B. and N. Flyer. A Primer on Radial Basis 
  Functions with Applications to the Geosciences. SIAM, 2015.
    
  '''
  x = np.asarray(x, dtype=float)
  assert_shape(x, (None,), 'x')
  
  s = np.asarray(s, dtype=float)
  assert_shape(s, (None, x.shape[0]), 's')
  
  diffs = np.asarray(diffs, dtype=int)
  diffs = _reshape_diffs(diffs)
  
  # stencil size and number of dimensions
  size, dim = s.shape
  if coeffs is None:
    coeffs = np.ones(diffs.shape[0], dtype=float)
  else:
    coeffs = np.asarray(coeffs, dtype=float)
    assert_shape(coeffs, (diffs.shape[0],), 'coeffs')

  phi = get_rbf(phi)

  max_order = _max_poly_order(size, dim)
  if order is None:
    order = _default_poly_order(diffs)
    order = min(order, max_order)

  if order > max_order:
    raise ValueError(
      'Polynomial order is too high for the stencil size')
    
  # get the powers for the added monomials
  pwr = powers(order, dim)
  # evaluate the RBF and monomials at each point in the stencil. This
  # becomes the left-hand-side
  A = phi(s, s, eps=eps)
  P = mvmonos(s, pwr)
  # Evaluate the RBF and monomials for each term in the differential
  # operator. This becomes the right-hand-side.
  a = coeffs[0]*phi(x[None, :], s, eps=eps, diff=diffs[0])
  p = coeffs[0]*mvmonos(x[None, :], pwr, diff=diffs[0])
  for c, d in zip(coeffs[1:], diffs[1:]):
    a += c*phi(x[None, :], s, eps=eps, diff=d)
    p += c*mvmonos(x[None, :], pwr, diff=d)

  # squeeze `a` and `p` into 1d arrays. `a` is ran through as_array
  # because it may be sparse.
  a = as_array(a)[0]
  p = p[0]

  # attempt to compute the RBF-FD weights
  try:
    w = PartitionedSolver(A, P).solve(a, p)[0]  
    return w

  except np.linalg.LinAlgError:
    raise np.linalg.LinAlgError(
      'An error was raised while computing the RBF-FD weights at '
      'point %s with the RBF %s and the polynomial order %s. This '
      'may be due to a stencil with duplicate or collinear points. '
      'The stencil contains the following points:\n%s' % 
      (x, phi, order, s))


def weight_matrix(x, p, n, diffs, coeffs=None,
                  phi=phs3, order=None,
                  eps=1.0, stencils=None):
  ''' 
  Returns a weight matrix which maps a functions values at `p` to an
  approximation of that functions derivative at `x`. This is a
  convenience function which first creates stencils and then computes
  the RBF-FD weights for each stencil.
  
  Parameters
  ----------
  x : (N, D) array
    Target points where the derivatives will be approximated. 

  p : (M, D) array
    Source points. The derivatives will be approximated with a
    weighted sum of values at these point.

  n : int
    The stencil size
  
  diffs : (D,) int array or (K, D) int array 
    Derivative orders for each spatial dimension. For example `[2, 0]`
    indicates that the weights should approximate the second
    derivative with respect to the first spatial dimension in
    two-dimensional space.  diffs can also be a (K, D) array, where
    each (D,) sub-array is a term in a differential operator. For
    example the two-dimensional Laplacian can be represented as 
    `[[2, 0], [0, 2]]`.

  coeffs : (K,) float array or (K, N) float, optional 
    Coefficients for each term in the differential operator specified
    with `diffs`. Defaults to an array of ones. If `diffs` was
    specified as a (D,) array then `coeffs` should be a length 1
    array. If the coefficients for the differential operator vary with
    `x` then `coeffs` can be specified as a (K, N) array.

  phi : rbf.basis.RBF, optional
    Type of RBF. Select from those available in `rbf.basis` or create 
    your own.

  order : int, optional
    Order of the added polynomial. This defaults to the highest
    derivative order. For example, if `diffs` is `[[2, 0], [0, 1]]`,
    then `order` is set to 2.

  eps : float or (M,) array, optional
    shape parameter for each RBF, which have centers `p`. This only 
    makes a difference when using RBFs that are not scale invariant.  
    All the predefined RBFs except for the odd order polyharmonic 
    splines are not scale invariant.

  Returns
  -------
  (N, M) coo sparse matrix          
      
  Examples
  --------
  Create a second order differentiation matrix in one-dimensional 
  space

  >>> x = np.arange(4.0)[:, None]
  >>> W = weight_matrix(x, x, 3, (2,))
  >>> W.toarray()
  array([[ 1., -2.,  1.,  0.],
         [ 1., -2.,  1.,  0.],
         [ 0.,  1., -2.,  1.],
         [ 0.,  1., -2.,  1.]])
                         
  '''
  x = np.asarray(x, dtype=float)
  assert_shape(x, (None, None), 'x')
  
  p = np.asarray(p, dtype=float)
  assert_shape(p, (None, x.shape[1]), 'p')
  
  diffs = np.asarray(diffs, dtype=int)
  diffs = _reshape_diffs(diffs)

  if np.isscalar(eps):
    eps = np.full(p.shape[0], eps, dtype=float)
  else:
    eps = np.asarray(eps, dtype=float)  
    assert_shape(eps, (p.shape[0],), 'eps')
    
  # make `coeffs` a (K, N) array
  if coeffs is None:
    coeffs = np.ones((diffs.shape[0], x.shape[0]), dtype=float)
  else:
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim == 1:
      coeffs = np.repeat(coeffs[:, None], x.shape[0], axis=1)

    assert_shape(coeffs, (diffs.shape[0], x.shape[0]), 'coeffs')

  stencils = KDTree(p).query(x, n)[1]

  logger.debug(
    'building a (%s, %s) RBF-FD weight matrix with %s nonzeros...' 
    % (x.shape[0], p.shape[0], stencils.size))   

  # values that will be put into the sparse matrix
  data = np.zeros(stencils.shape, dtype=float)
  for i, si in enumerate(stencils):
    # intermittently log the progress 
    if i % max(stencils.shape[0] // 10, 1) == 0:
      logger.debug('  %d%% complete' % (100*i / stencils.shape[0]))

    data[i, :] = weights(x[i], p[si], diffs,
                         coeffs=coeffs[:, i], eps=eps[si],
                         phi=phi, order=order)

    
  rows = np.repeat(range(data.shape[0]), data.shape[1])
  cols = stencils.ravel()
  data = data.ravel()
  shape = x.shape[0], p.shape[0]
  L = sp.coo_matrix((data, (rows, cols)), shape)
  logger.debug('  done')
  return L
