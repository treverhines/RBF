'''
This module provides classes for RBF interpolation, `RBFInterpolant` and
`NearestRBFInterpolant`. The latter is more suitable when there are a large
number of observations because it performs RBF interpolation using only the k
nearest observations to each interpolation point.

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
:math:`\mathbf{P(x)}` is a Vandermonde matrix containing :math:`p` monomial
basis functions evaluated at the interpolation points. The monomial basis
functions span the space of all polynomials with a user specified degree.
The coefficients :math:`\mathbf{a}` and :math:`\mathbf{b}` are chosen to
minimize the objective function

.. math::
  \mathcal{L}(\mathbf{a, b}) =
  \mathbf{a^T K(y,y) a} +
  \lambda \mathbf{(f(y; a, b) - d)^T \Sigma^{-1} (f(y; a, b) - d)}.


For the minimization problem to be well-posed, we assume that the RBF is
positive definite or conditionally positive definite of order :math:`m` (see
Chapter 7 of [1]). If the RBF is conditionally positive definite of order
:math:`m`, then we assume the degree of the added polynomial is
:math:`\ge(m-1)`.

The first term in the above objective function is the native space norm
associated with the RBF (See Chapter 13 of [1]). Loosely speaking, it can be
viewed as a measure of the roughness of the interpolant. The second term in the
objective function measures the misfit between the interpolant and the
observations. We have introduced the parameter :math:`\lambda` to control the
trade-off between these two terms.

To determine :math:`\mathbf{a}` and :math:`\mathbf{b}`, we need :math:`n+p`
linear constraints. If we first assume that the RBF is positive definite (and
therefore :math:`\mathbf{K(y, y)}` is positive definite), then we can recognize
that :math:`\mathcal{L}` is a convex function, and so it can be minimized by
finding where its gradient is zero. We differentiate :math:`\mathcal{L}` with
respect to :math:`\mathbf{a}` and set it equal to zero to get :math:`n`
constraints

.. math::
  (\mathbf{K(y,y)} + \lambda^{-1}\mathbf{\Sigma}) \mathbf{a}
  + \mathbf{P(y) b} = \mathbf{d}.

For the remaining :math:`p` constraints, we differentiate
:math:`\mathcal{L}` with respect to :math:`\mathbf{b}`, set it equal to zero,
and substitute the above equation in for :math:`\mathbf{d}` to get

.. math::
  \mathbf{P(y)^T a} = \mathbf{0}.


The above two equations then provide us with the linear constraints needed to
solve for :math:`\mathbf{a}` and :math:`\mathbf{b}`. In the case that the RBF
is conditionally positive definite, we want to minimize the objective function
subject to the imposed constraint that :math:`\mathbf{P(y)^T a} = \mathbf{0}`,
which ensures that :math:`\mathbf{a^T K(y, y) a} \ge 0`. We then arrive at the
same solution for :math:`\mathbf{a}` and :math:`\mathbf{b}`.

There are circumstances when the above two equations are not sufficient to
solve for :math:`\mathbf{a}` and :math:`\mathbf{b}`. For example, there may not
be a unique solution if observations are colinear or coplanar. It is also
necessary to have at least :math:`p` observations. The user is referred to [1]
for details on when the interpolation problem is well-posed.

The above formulation for the RBF interpolant can be found in section 19.2 and
6.3 of [1]. It can also be found in the context of thin-plate splines in
section 2.4 of [2] and in the context of Kriging in section 3.4 of [3].

In our implementation, we have combined the effect of :math:`\Sigma` and
:math:`\lambda` into one variable, `sigma`, which is a scalar or a vector that
is *proportional* to the standard deviation of the noise. When `sigma` is `0`
the observations are fit perfectly by the interpolant. Increasing `sigma`
degrades the fit while improving the smoothness of the interpolant.

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
from rbf.poly import monomial_count, mvmonos
from rbf.basis import get_rbf, SparseRBF
from rbf.utils import assert_shape, KDTree
from rbf.linalg import PartitionedSolver

logger = logging.getLogger(__name__)


# Interpolation with conditionally positive definite RBFs has no assurances of
# being well posed when the order of the added polynomial is not high enough.
# Define that minimum polynomial order here. These values are from Chapter 8 of
# Fasshauer's "Meshfree Approximation Methods with MATLAB"
_MIN_ORDER = {
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
               phi='phs3',
               eps=1.0,
               order=None):
    y = np.asarray(y, dtype=float)
    assert_shape(y, (None, None), 'y')
    ny, ndim = y.shape

    d = np.asarray(d, dtype=float)
    assert_shape(d, (ny,), 'd')

    if np.isscalar(sigma):
      sigma = np.full(ny, sigma, dtype=float)
    else:
      sigma = np.asarray(sigma, dtype=float)
      assert_shape(sigma, (ny,), 'sigma')

    phi = get_rbf(phi)

    if not np.isscalar(eps):
      logger.warning(
        'The shape parameter should be a float in order for the interpolant '
        'to be well-posed')

    # If the `phi` is not in `_MIN_ORDER`, then the RBF is either
    # positive definite (no minimum polynomial order) or user-defined
    min_order = _MIN_ORDER.get(phi, -1)
    if order is None:
      order = max(min_order, 0)
    elif order < min_order:
      logger.warning(
        'The polynomial order should not be below %d for %s in order for the '
        'interpolant to be well-posed' % (min_order, phi))

    # Build the system of equations and solve for the RBF and mononomial
    # coefficients
    Kyy = phi(y, y, eps=eps)
    S = scipy.sparse.diags(sigma**2)
    Py = mvmonos(y, order)
    nmonos = Py.shape[1]
    if nmonos > ny:
      raise ValueError(
        'The polynomial order is too high. The number of monomials, %d, '
        'exceeds the number of observations, %d' % (nmonos, ny))

    z = np.zeros(nmonos, dtype=float)
    phi_coeff, poly_coeff = PartitionedSolver(Kyy + S, Py).solve(d, z)

    self.y = y
    self.phi = phi
    self.eps = eps
    self.order = order
    self.phi_coeff = phi_coeff
    self.poly_coeff = poly_coeff

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
    nx = x.shape[0]

    if chunk_size is not None:
      out = np.zeros(nx, dtype=float)
      for start in range(0, nx, chunk_size):
        stop = start + chunk_size
        out[start:stop] = self(x[start:stop], diff=diff, chunk_size=None)

      return out

    Kxy = self.phi(x, self.y, eps=self.eps, diff=diff)
    Px = mvmonos(x, self.order, diff=diff)
    out = Kxy.dot(self.phi_coeff) + Px.dot(self.poly_coeff)
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
               phi='phs3',
               eps=1.0,
               order=None):
    y = np.asarray(y, dtype=float)
    assert_shape(y, (None, None), 'y')
    ny, ndim = y.shape

    d = np.asarray(d, dtype=float)
    assert_shape(d, (ny,), 'd')

    if np.isscalar(sigma):
      sigma = np.full(ny, sigma, dtype=float)
    else:
      sigma = np.asarray(sigma, dtype=float)
      assert_shape(sigma, (ny,), 'sigma')

    # make sure the number of nearest neighbors used for interpolation does not
    # exceed the number of observations
    k = min(int(k), ny)

    phi = get_rbf(phi)
    if isinstance(phi, SparseRBF):
      raise ValueError('SparseRBF instances are not supported')

    if not np.isscalar(eps):
      raise ValueError('The shape parameter should be a float')

    min_order = _MIN_ORDER.get(phi, -1)
    if order is None:
      order = max(min_order, 0)
    elif order < min_order:
      logger.warning(
        'The polynomial order should not be below %d for %s in order for the '
        'interpolant to be well-posed' % (min_order, phi))

    nmonos = monomial_count(order, ndim)
    if nmonos > k:
      raise ValueError(
        'The polynomial order is too high. The number of monomials, %d, '
        'exceeds the number of neighbors used for interpolation, %d' %
        (nmonos, k))

    tree = KDTree(y)

    self.y = y
    self.d = d
    self.sigma = sigma
    self.k = k
    self.eps = eps
    self.phi = phi
    self.order = order
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
    nx = x.shape[0]

    if chunk_size is not None:
      out = np.zeros(nx, dtype=float)
      for start in range(0, nx, chunk_size):
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
    nnbr = nbr.shape[0]

    # build the left-hand-side interpolation matrix consisting of the RBF
    # and polyomials evaluated at each neighborhood
    Kyy = self.phi(self.y[nbr], self.y[nbr], eps=self.eps)
    Kyy[:, range(self.k), range(self.k)] += self.sigma[nbr]**2
    Py = mvmonos(self.y[nbr], self.order)
    Pyt = np.transpose(Py, (0, 2, 1))
    nmonos = Py.shape[2]
    Z = np.zeros((nnbr, nmonos, nmonos), dtype=float)
    LHS = np.block([[Kyy, Py], [Pyt, Z]])

    # build the right-hand-side data vector consisting of the observations for
    # each neighborhood and extra zeros
    z = np.zeros((nnbr, nmonos), dtype=float)
    rhs = np.hstack((self.d[nbr], z))

    # solve for the RBF and polynomial coefficients for each neighborhood
    coeff = np.linalg.solve(LHS, rhs)
    # expand the coefficients and neighborhoods so that there is one set for
    # each interpolation point
    coeff = coeff[inv]
    nbr = nbr[inv]

    # evaluate the interpolant at the interpolation points
    phi_coeff = coeff[:, :self.k]
    poly_coeff = coeff[:, self.k:]
    Kxy = self.phi(x[:, None], self.y[nbr], eps=self.eps, diff=diff)[:, 0]
    Px = mvmonos(x, self.order, diff=diff)
    out = (Kxy*phi_coeff).sum(axis=1) + (Px*poly_coeff).sum(axis=1)
    return out
