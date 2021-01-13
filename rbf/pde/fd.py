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
from rbf.linalg import as_array

logger = logging.getLogger(__name__)


@Memoize
def _max_poly_order(size, dim):
  '''
  Returns the maximum polynomial order allowed for the given stencil size and
  number of dimensions
  '''
  order = -1
  while (count(order+1, dim) <= size):
    order += 1

  return order


def weights(x, s, diffs,
            coeffs=None,
            phi=phs3,
            order=None,
            eps=1.0):
  '''
  Returns the weights which map a function's values at `s` to an approximation
  of that function's derivative at `x`. The weights are computed using the
  RBF-FD method described in [1]

  Parameters
  ----------
  x : (..., D) float array
    Target points where the derivative is being approximated

  s : (..., M, D) float array
    Stencils for each target point

  diffs : (D,) int array or (K, D) int array
    Derivative orders for each spatial dimension. For example `[2, 0]`
    indicates that the weights should approximate the second derivative with
    respect to the first spatial dimension in two-dimensional space.  `diffs`
    can also be a (K, D) array, where each (D,) sub-array is a term in a
    differential operator. For example, the two-dimensional Laplacian can be
    represented as `[[2, 0], [0, 2]]`.

  coeffs : (K, ...) float array, optional
    Coefficients for each term in the differential operator specified with
    `diffs`. The coefficients can vary between target points. Defaults to an
    array of ones.

  phi : rbf.basis.RBF instance or str, optional
    Type of RBF. See `rbf.basis` for the available options.

  order : int, optional
    Order of the added polynomial. This defaults to the highest derivative
    order. For example, if `diffs` is `[[2, 0], [0, 1]]`, then this is set to
    2.

  eps : float or float array, optional
    Shape parameter

  Returns
  -------
  (..., M) float array
    RBF-FD weights for each target point

  Examples
  --------
  Calculate the weights for a one-dimensional second order derivative.

  >>> x = np.array([1.0])
  >>> s = np.array([[0.0], [1.0], [2.0]])
  >>> diff = (2,)
  >>> weights(x, s, diff)
  array([ 1., -2., 1.])

  Calculate the weights for estimating an x derivative from three points in a
  two-dimensional plane

  >>> x = np.array([0.25, 0.25])
  >>> s = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]])
  >>> diff = (1, 0)
  >>> weights(x, s, diff)
  array([ -1., 1., 0.])

  References
  ----------
  [1] Fornberg, B. and N. Flyer. A Primer on Radial Basis Functions with
  Applications to the Geosciences. SIAM, 2015.

  '''
  x = np.asarray(x, dtype=float)
  assert_shape(x, (..., None), 'x')
  bcast = x.shape[:-1]
  ndim = x.shape[-1]

  s = np.asarray(s, dtype=float)
  assert_shape(s, (..., None, ndim), 's')
  # broadcast leading dimensions of `s` to match leading dimensions of `x`
  s = np.broadcast_to(s, bcast + s.shape[-2:])
  ssize = s.shape[-2]

  diffs = np.asarray(diffs, dtype=int)
  diffs = np.atleast_2d(diffs)
  assert_shape(diffs, (None, ndim), 'diffs')

  if coeffs is None:
    coeffs = np.ones(len(diffs), dtype=float)
  else:
    coeffs = np.asarray(coeffs, dtype=float)
    assert_shape(coeffs, (len(diffs), ...), 'coeffs')

  # broadcast each element in `coeffs` to match leading dimensions of `x`
  coeffs = [np.broadcast_to(c, bcast) for c in coeffs]

  phi = get_rbf(phi)

  # get the maximum polynomial order allowed for this stencil size
  max_order = _max_poly_order(ssize, ndim)
  if order is None:
    # If the polynomial order is not specified, make it equal to the derivative
    # order, provided that the stencil size is large enough.
    order = diffs.sum(axis=1).max()
    order = min(order, max_order)

  if order > max_order:
    raise ValueError('Polynomial order is too high for the stencil size')

  # center the stencil on `x` for improved numerical stability
  x = x[..., None, :]
  s = s - x
  x = np.zeros_like(x)
  # get the powers for the added monomials
  pwr = powers(order, ndim)
  # evaluate the RBF and monomials at each point in the stencil. This becomes
  # the left-hand-side
  A = phi(s, s, eps=eps)
  P = mvmonos(s, pwr)
  Pt = np.einsum('...ij->...ji', P)
  Z = np.zeros(bcast + (len(pwr), len(pwr)), dtype=float)
  LHS = np.concatenate(
    (np.concatenate((A, P), axis=-1),
     np.concatenate((Pt, Z), axis=-1)),
    axis=-2)

  # Evaluate the RBF and monomials at the target points for each term in the
  # differential operator. This becomes the right-hand-side.
  a, p = 0.0, 0.0
  for c, d in zip(coeffs, diffs):
    a += c[..., None, None]*phi(x, s, eps=eps, diff=d)
    p += c[..., None, None]*mvmonos(x, pwr, diff=d)

  # convert `a` to an array because phi may be a sparse RBF
  a = as_array(a)[..., 0, :]
  p = p[..., 0, :]
  rhs = np.concatenate((a, p), axis=-1)

  w = np.linalg.solve(LHS, rhs)[..., :ssize]
  return w


def weight_matrix(x, p, n, diffs, *args, **kwargs):
  '''
  Returns a weight matrix which maps a function's values at `p` to an
  approximation of that function's derivative at `x`. This is a convenience
  function which first creates stencils and then computes the RBF-FD weights
  for each stencil.

  Parameters
  ----------
  x : (N, D) float array
    Target points where the derivative is being approximated
n
  p : (M, D) array
    Source points. The derivatives will be approximated with a weighted sum of
    values at these point.

  n : int
    The stencil size. Each target point will have a stencil made of the `n`
    nearest neighbors from `p`

  diffs : (D,) int array or (K, D) int array
    Derivative orders for each spatial dimension. For example `[2, 0]`
    indicates that the weights should approximate the second derivative with
    respect to the first spatial dimension in two-dimensional space.  `diffs`
    can also be a (K, D) array, where each (D,) sub-array is a term in a
    differential operator. For example the two-dimensional Laplacian can be
    represented as `[[2, 0], [0, 2]]`.

  coeffs : (K,) or (K, N) float array, optional
    Coefficients for each term in the differential operator specified with
    `diffs`. The coefficients can vary between target points. Defaults to an
    array of ones.

  phi : rbf.basis.RBF instance or str, optional
    Type of RBF. Select from those available in `rbf.basis` or create your own.

  order : int, optional
    Order of the added polynomial. This defaults to the highest derivative
    order. For example, if `diffs` is `[[2, 0], [0, 1]]`, then this is set to
    2.

  eps : float or float array, optional
    Shape parameter. This can be a float or an array that is broadcastable to
    `s`

  Returns
  -------
  (N, M) coo sparse matrix

  Examples
  --------
  Create a second order differentiation matrix in one-dimensional space

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

  _, stencils = KDTree(p).query(x, n)

  data = weights(x, p[stencils], diffs, *args, **kwargs)
  data = data.ravel()
  rows = np.repeat(range(len(x)), n)
  cols = stencils.ravel()
  out = sp.coo_matrix((data, (rows, cols)), (len(x), len(p)))
  return out
