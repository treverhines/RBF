'''
This module provides a class for RBF interpolation, `RBFInterpolant`. This
function has numerous features that are lacking in `scipy.interpolate.rbf`.
They include:

* variable weights on the data (when creating a smoothed interpolant)
* more choices of basis functions (you can also easily make your own)
* analytical differentiation of the interpolant
* added polynomial terms for improved accuracy

RBF Interpolation
-----------------
Consider a vector of :math:`n` observations :math:`\mathbf{d}` made at
locations :math:`\mathbf{y}`. These observations could potentially have
normally distributed noise described by the covariance matrix :math:`\Sigma`.
An RBF interpolant :math:`\mathbf{f(x)}` for these observations is
parameterized as

.. math::
  \mathbf{f(x)} = \mathbf{K(x,y) a} + \mathbf{P(x) b}

where :math:`\mathbf{K(x,y)}` consists of the RBFs with centers at
:math:`\mathbf{y}` evaluated at the interpolation points :math:`\mathbf{x}`.
:math:`\mathbf{P(x)}` is a polynomial matrix containing :math:`m` monomial
basis function evaluated at the interpolation points. The monomial basis
functions span the space of all polynomials with a user specified order. The
coefficients :math:`\mathbf{a}` and :math:`\mathbf{b}` are chosen to minimize
the objective function

.. math::
  \mathcal{L}(\mathbf{a, b}) =
  \mathbf{a^T K(y,y) a} +
  \lambda \mathbf{(f(y; a, b) - d)^T \Sigma^{-1} (f(y; a, b) - d)}.

In the above expression, the second term on the right side is a norm measuring
the misfit between the interpolant and the observations. The first term on the
right side is a norm that essentially penalizes the roughness of the
interpolant (technically, it is the norm associated with the reproducing kernel
Hilbert space for the chosen radial basis function). We have also introduced
:math:`\lambda` which is a smoothing parameter that controls how well we want
the interpolant to fit the data.

To determine :math:`\mathbf{a}` and :math:`\mathbf{b}`, we need :math:`n+m`
linear constraints. We can recognize that :math:`\mathcal{L}` is a convex
function, and so it can be minimized by finding where its gradient is zero. We
differentiate :math:`\mathcal{L}` with respect to :math:`\mathbf{a}` and set it
equal to zero to get :math:`n` constraints


.. math::
  (\mathbf{K(y,y)} + \lambda^{-1}\mathbf{\Sigma}) \mathbf{a}
  + \mathbf{P(y) b} = \mathbf{d}.

For the remaining :math:`m` constraints, we differentiate
:math:`\mathcal{L}` with respect to :math:`\mathbf{b}`, set it equal to zero,
and substitute the above equation in for :math:`\mathbf{d}` to get

.. math::
  \mathbf{P(y)^T a} = \mathbf{0}.

For most purposes, the above two equations provide the linear constrains that
are needed to solve for :math:`\mathbf{a}` and :math:`\mathbf{b}`. However,
there are cases where the system cannot be solved due to, for example, too few
data points or a polynomial order that is too high. There are also some radial
basis functions that are conditionally positive definite, which means that the
norm :math:`\mathbf{a^T K(y, y) a}` is only guaranteed to be positive when the
order of the added polynomial is sufficiently high. The user is referred to [1]
for details on when the interpolation problem is well-posed.

In our implementation, we have combined the effect of :math:`\Sigma` and
:math:`\lambda` into one variable, `sigma`, which is a scalar or a vector that
is *proportional* to the standard deviation of the noise. When `sigma` is `0`
the observations are fit perfectly by the interpolant. Increasing `sigma`
degrades the fit while improving the smoothness of the interpolant.

The above formulation for the RBF interpolant can be found in section 19.2 and
6.3 of [1]. It can also be found in the context of thin-plate splines in
section 2.4 of [2] and in the context of Kriging in section 3.4 of [3].

References
----------
[1] Fasshauer, G., Meshfree Approximation Methods with Matlab. World
Scientific Publishing Co, 2007.

[2] Wahba, G., Spline Models for Observational Data. SIAM, 1990

[3] Cressie, N., Statistics for Spatial Data. John Wiley & Sons, Inc, 1993

'''
import logging

import numpy as np
import scipy.sparse
import scipy.spatial

import rbf.basis
from rbf.poly import powers, mvmonos
from rbf.basis import get_rbf, SparseRBF
from rbf.utils import assert_shape, KDTree
from rbf.linalg import PartitionedSolver

logger = logging.getLogger(__name__)


def _in_hull(p, hull):
  '''
  Tests if points in `p` are in the convex hull made up by `hull`
  '''
  dim = p.shape[1]
  # if there are not enough points in `hull` to form a simplex then
  # return False for each point in `p`.
  if len(hull) <= dim:
    return np.zeros(len(p), dtype=bool)

  if dim >= 2:
    hull = scipy.spatial.Delaunay(hull)
    return hull.find_simplex(p)>=0
  else:
    # one dimensional points
    min = np.min(hull)
    max = np.max(hull)
    return (p[:, 0] >= min) & (p[:, 0] <= max)


# Interpolation with conditionally positive definite RBFs has no assurances of
# being well posed when the order of the added polynomial is not high enough.
# Define that minimum polynomial order here. These values are from Chapter 8 of
# Fasshauer's "Meshfree Approximation Methods with MATLAB"
_RECOMMENDED_ORDER = {
  rbf.basis.mq: 0,
  rbf.basis.phs1: 0,
  rbf.basis.phs2: 1,
  rbf.basis.phs3: 1,
  rbf.basis.phs4: 2,
  rbf.basis.phs5: 2,
  rbf.basis.phs6: 3,
  rbf.basis.phs7: 3,
  rbf.basis.phs8: 4
  }


class RBFInterpolant(object):
  '''
  Regularized radial basis function interpolant

  Parameters
  ----------
  y : (N, D) array
    Observation points

  d : (N,) array
    Observed values at `y`

  sigma : float or (N,) array, optional
    Smoothing parameter. Setting this to 0 causes the interpolant to perfectly
    fit the data. Increasing the smoothing parameter degrades the fit while
    improving the smoothness of the interpolant. If this is a vector, it should
    be proportional to the one standard deviation uncertainties for the
    observations. This defaults to zeros.

  eps : float, optional
    Shape parameter

  phi : rbf.basis.RBF instance or str, optional
    Radial basis function. This can be an `RBF` instance (e.g,
    `rbf.basis.phs3`) or a string for a predefined RBF (e.g., 'phs3'). See
    `rbf.basis` for all the available options.

  order : int, optional
    Order of the added polynomial terms. Set this to `-1` for no added
    polynomial terms.

  extrapolate : bool, optional
    Whether to allows points to be extrapolated outside of a convex hull formed
    by `y`. If False, then np.nan is returned for outside points.

  Notes
  -----
  * If `phi` is conditionally positive definite of order `i` (see `rbf.basis`),
    then the polynomial order should be at least `i - 1` to provide some
    assurance that the interpolation problem is well posed (see Chapters 7 and
    8 of [1]). For example, if `phi` is "phs1", "phs2", or "phs3", then `order`
    should be at least `0`, `1`, or `1`, respectively.

  References
  ----------
  [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World
  Scientific Publishing Co, 2007.

  '''
  def __init__(self, y, d,
               sigma=0.0,
               eps=1.0,
               phi='phs3',
               order=None,
               extrapolate=True):
    y = np.asarray(y, dtype=float)
    assert_shape(y, (None, None), 'y')

    d = np.asarray(d, dtype=float)
    assert_shape(d, (len(y),), 'd')

    if np.isscalar(sigma):
      sigma = np.full(len(y), sigma, dtype=float)
    else:
      sigma = np.asarray(sigma, dtype=float)
      assert_shape(sigma, (len(y),), 'sigma')

    if not np.isscalar(eps):
      logger.warning(
        'The shape parameter should be a float in order for the interpolant '
        'to be well-posed')

    phi = get_rbf(phi)
    # If the `phi` is not in `_RECOMMENDED_ORDER`, then the RBF is either
    # positive definite (no minimum polynomial order) or user-defined
    if order is None:
      order = _RECOMMENDED_ORDER.get(phi, 0)

    recommended_order = _RECOMMENDED_ORDER.get(phi, -1)
    if order < recommended_order:
      logger.warning(
        'The polynomial order should not be below %d for %s in order for the '
        'interpolant to be well-posed' % (recommended_order, phi))

    # get the powers for each monomial in the added polynomial. Make sure there
    # are not more monomials than observations
    pwr = powers(order, y.shape[1])
    if len(pwr) > len(y):
      raise ValueError(
        'The polynomial order is too high for the number of observations')

    # Build the system of equations and solve for the RBF and mononomial
    # coefficients
    Kyy = phi(y, y, eps=eps)
    S = scipy.sparse.diags(sigma**2)
    Py = mvmonos(y, pwr)
    z = np.zeros(len(pwr), dtype=float)
    phi_coeff, poly_coeff = PartitionedSolver(Kyy + S, Py).solve(d, z)

    self.y = y
    self.phi = phi
    self.order = order
    self.eps = eps
    self.phi_coeff = phi_coeff
    self.poly_coeff = poly_coeff
    self.pwr = pwr
    self.extrapolate = extrapolate

  def __call__(self, x, diff=None, chunk_size=1000):
    '''
    Evaluates the interpolant at `x`

    Parameters
    ----------
    x : (N, D) float array
      Target points

    diff : (D,) int array, optional
      Derivative order for each spatial dimension

    chunk_size : int, optional
      Break `x` into chunks with this size and evaluate the interpolant for
      each chunk

    Returns
    -------
    (N,) float array

    '''
    x = np.asarray(x,dtype=float)
    assert_shape(x, (None, self.y.shape[1]), 'x')

    if chunk_size is not None:
      out = np.zeros(len(x), dtype=float)
      for start in range(0, len(x), chunk_size):
        stop = start + chunk_size
        out[start:stop] = self(x[start:stop], diff=diff, chunk_size=None)

      return out

    Kxy = self.phi(x, self.y, eps=self.eps, diff=diff)
    Px = mvmonos(x, self.pwr, diff=diff)
    out = Kxy.dot(self.phi_coeff) + Px.dot(self.poly_coeff)

    # return nan for points outside of the convex hull if extrapolation is not
    # allowed
    if not self.extrapolate:
      out[~_in_hull(x, self.y)] = np.nan

    return out


class NearestRBFInterpolant(object):
  '''
  Approximation to `RBFInterpolant` that only uses the k nearest observations
  for each interpolation point.

  Parameters
  ----------
  y : (N, D) array
    Observation points

  d : (N,) array
    Observed values at `y`

  sigma : float or (N,) array, optional
    Smoothing parameter. Setting this to 0 causes the interpolant to perfectly
    fit the data. Increasing the smoothing parameter degrades the fit while
    improving the smoothness of the interpolant. If this is a vector, it should
    be proportional to the one standard deviation uncertainties for the
    observations. This defaults to zeros.

  k : int, optional
    Number of neighboring observations to use for each interpolation point

  eps : float, optional
    Shape parameter

  phi : rbf.basis.RBF instance or str, optional
    Radial basis function. This can be an `RBF` instance (e.g,
    `rbf.basis.phs3`) or a string for a predefined RBF (e.g., 'phs3'). See
    `rbf.basis` for all the available options. `SparseRBF` instances are not
    supported.

  order : int, optional
    Order of the added polynomial terms. Set this to `-1` for no added
    polynomial terms.

  Notes
  -----
  * If `phi` is conditionally positive definite of order `i` (see `rbf.basis`),
    then the polynomial order should be at least `i - 1` to provide some
    assurance that the interpolation problem is well-posed (see section 7.1 of
    [1]). For example, if `phi` is "phs1", "phs2", or "phs3", then `order`
    should be at least `0`, `1`, or `1`, respectively.

  References
  ----------
  [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World
  Scientific Publishing Co, 2007.

  '''
  def __init__(self, y, d,
               sigma=0.0,
               k=20,
               eps=1.0,
               phi='phs3',
               order=None):
    y = np.asarray(y, dtype=float)
    assert_shape(y, (None, None), 'y')

    d = np.asarray(d, dtype=float)
    assert_shape(d, (len(y),), 'd')

    if np.isscalar(sigma):
      sigma = np.full(len(y), sigma, dtype=float)
    else:
      sigma = np.asarray(sigma, dtype=float)
      assert_shape(sigma, (len(y),), 'sigma')

    # make sure the number of nearest neighbors used for interpolation does not
    # exceed the number of observations
    k = min(k, len(y))

    if not np.isscalar(eps):
      raise ValueError('The shape parameter should be a float')

    phi = get_rbf(phi)
    if isinstance(phi, SparseRBF):
      raise ValueError('SparseRBF instances are not supported')

    # If the `phi` is not in `_RECOMMENDED_ORDER`, then the RBF is either
    # positive definite (no minimum polynomial order) or user-defined
    if order is None:
      order = _RECOMMENDED_ORDER.get(phi, 0)

    recommended_order = _RECOMMENDED_ORDER.get(phi, -1)
    if order < recommended_order:
      logger.warning(
        'The polynomial order should not be below %d for %s in order for the '
        'interpolant to be well-posed' % (recommended_order, phi))

    pwr = powers(order, y.shape[1])
    if len(pwr) > k:
      raise ValueError(
        'The polynomial order is too high for the number of nearest neighbors')

    tree = KDTree(y)

    self.y = y
    self.d = d
    self.sigma = sigma
    self.k = k
    self.eps = eps
    self.phi = phi
    self.pwr = pwr
    self.tree = tree

  def __call__(self, x, diff=None, chunk_size=100):
    '''
    Evaluates the interpolant at `x`

    Parameters
    ----------
    x : (N, D) float array
      Target points

    diff : (D,) int array, optional
      Derivative order for each spatial dimension

    chunk_size : int, optional
      Break `x` into chunks with this size and evaluate the interpolant for
      each chunk

    Returns
    -------
    (N,) float array

    '''
    x = np.asarray(x, dtype=float)
    assert_shape(x, (None, self.y.shape[1]), 'x')

    if chunk_size is not None:
      out = np.zeros(len(x), dtype=float)
      for start in range(0, len(x), chunk_size):
        stop = start + chunk_size
        out[start:stop] = self(x[start:stop], diff=diff, chunk_size=None)

      return out

    # get the indices of the k nearest observations for each interpolation
    # point
    _, nbr = self.tree.query(x, self.k)
    # multiple interpolation points may have the same neighborhood. Make the
    # neighborhoods unique so that we only compute the interpolation
    # coefficients once for each neighborhood
    nbr, inv = np.unique(np.sort(nbr, axis=1), return_inverse=True, axis=0)

    # build the left-hand-side interpolation matrix consisting of the RBF
    # and polyomials evaluated at each neighborhood
    Kyy = self.phi(self.y[nbr], self.y[nbr], eps=self.eps)
    # add the smoothing term to the diagonals of Kyy
    Kyy[:, range(self.k), range(self.k)] += self.sigma[nbr]**2
    Py = mvmonos(self.y[nbr], self.pwr)
    Pyt = np.einsum('ijk->ikj', Py)
    Z = np.zeros((len(nbr), len(self.pwr), len(self.pwr)), dtype=float)
    LHS = np.concatenate(
      (np.concatenate((Kyy, Py), axis=2),
       np.concatenate((Pyt, Z), axis=2)),
      axis=1)

    # build the right-hand-side data vector consisting of the observations for
    # each neighborhood and extra zeros
    z = np.zeros((len(nbr), len(self.pwr)), dtype=float)
    rhs = np.concatenate((self.d[nbr], z), axis=1)

    # solve for the RBF and polynomial coefficients
    coeff = np.linalg.solve(LHS, rhs)
    # expand the coefficients and neighborhoods so that there is one set for
    # each interpolation point
    coeff = coeff[inv]
    nbr = nbr[inv]

    # evaluate the interpolant at the interpolation points
    phi_coeff = coeff[:, :self.k]
    poly_coeff = coeff[:, self.k:]
    Kxy = self.phi(x[:, None], self.y[nbr], eps=self.eps, diff=diff)[:, 0]
    Px = mvmonos(x, self.pwr, diff=diff)
    out = (Kxy*phi_coeff).sum(axis=1) + (Px*poly_coeff).sum(axis=1)
    return out
