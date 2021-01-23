'''
This module provides classes for RBF interpolation, `RBFInterpolant` and
`NearestRBFInterpolant`. The latter is more suitable when there are a large
number of observations because it performs RBF interpolation using only the `k`
nearest observations to each interpolation point.

RBF Interpolation
-----------------
An RBF interpolant fits scalar valued observations
:math:`\mathbf{d}=[d_1,...,d_n]^T` made at the scattered locations
:math:`y_1,...,y_n`. The RBF interpolant is parameterized as

.. math::
  f(x) = \sum_{i=1}^n a_i \phi(||x - y_i||_2) +
         \sum_{j=1}^m b_j p_j(x)

where :math:`\phi` is an RBF and :math:`p_1(x),...,p_m(x)` are monomials that
span the space of polynomials with a specified degree. The coefficients
:math:`\mathbf{a}=[a_1,...,a_n]^T` and :math:`\mathbf{b}=[b_1,...,b_m]^T`
are the solutions to the linear equations

.. math::
    (\mathbf{K} + \sigma^2\mathbf{I})\mathbf{a} + \mathbf{P b} = \mathbf{d}

and

.. math::
    \mathbf{P^T a} = \mathbf{0},

where :math:`K_{ij} = \phi(||y_i - y_j||_2)`, :math:`P_{ij}=p_j(y_i)`, and
:math:`\sigma` is a smoothing parameter that controls how well we want to fit
the observations. The observations are fit exactly when :math:`\sigma` is
zero.

If the chosen RBF is positive definite (see `rbf.basis`) and :math:`\mathbf{P}`
has full rank, the solution for :math:`\mathbf{a}` and :math:`\mathbf{b}` is
unique. If the chosen RBF is conditionally positive definite of order `q` and
:math:`\mathbf{P}` has full rank, the solution is unique provided that the
degree of the monomial terms is at least `q-1` (see Chapter 7 of [1] or [2]).

References
----------
[1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab. World
Scientific Publishing Co.

[2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

[3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

[4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

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
    The type of RBF. This can be an `rbf.basis.RBF` instance or the RBF
    abbreviation as a string. See `rbf.basis` for the available options.

  order : int, optional
    Order of the added polynomial terms. Set this to `-1` for no added
    polynomial terms. If `phi` is a conditionally positive definite RBF of
    order `m`, then this value should be at least `m - 1`.

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
      raise ValueError('The shape parameter should be a float')

    # If `phi` is not in `_MIN_ORDER`, then the RBF is either positive definite
    # (no minimum polynomial order) or user-defined
    min_order = _MIN_ORDER.get(phi, -1)
    if order is None:
      order = max(min_order, 0)
    elif order < min_order:
      logger.warning(
        'The polynomial order should not be below %d for %s in order for the '
        'interpolant to be well-posed' % (min_order, phi))

    order = int(order)
    # For improved numerical stability, shift the observations so that their
    # centroid is at zero
    center = y.mean(axis=0)
    y = y - center
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
    self.center = center
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

    x = x - self.center
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
    The type of RBF. This can be an `rbf.basis.RBF` instance or the RBF
    abbreviation as a string. See `rbf.basis` for the available options.

  order : int, optional
    Order of the added polynomial terms. Set this to -1 for no added polynomial
    terms. If `phi` is a conditionally positive definite RBF of order `m`, then
    this value should be at least `m - 1`.

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

    order = int(order)
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

    # get the indices of the k-nearest observations for each interpolation
    # point
    _, nbr = self.tree.query(x, self.k)
    # multiple interpolation points may have the same neighborhood. Make the
    # neighborhoods unique so that we only compute the interpolation
    # coefficients once for each neighborhood
    nbr, inv = np.unique(np.sort(nbr, axis=1), return_inverse=True, axis=0)
    nnbr = nbr.shape[0]
    # Get the observation data for each neighborhood
    y, d, sigma = self.y[nbr], self.d[nbr], self.sigma[nbr]
    # shift the centers of each neighborhood to zero for numerical stability
    centers = y.mean(axis=1)
    y = y - centers[:, None]
    # build the left-hand-side interpolation matrix consisting of the RBF
    # and monomials evaluated at each neighborhood
    Kyy = self.phi(y, y, eps=self.eps)
    Kyy[:, range(self.k), range(self.k)] += sigma**2
    Py = mvmonos(y, self.order)
    PyT = np.transpose(Py, (0, 2, 1))
    nmonos = Py.shape[2]
    Z = np.zeros((nnbr, nmonos, nmonos), dtype=float)
    LHS = np.block([[Kyy, Py], [PyT, Z]])
    # build the right-hand-side data vector consisting of the observations for
    # each neighborhood and extra zeros
    z = np.zeros((nnbr, nmonos), dtype=float)
    rhs = np.hstack((d, z))
    # solve for the RBF and polynomial coefficients for each neighborhood
    coeff = np.linalg.solve(LHS, rhs)
    # expand the arrays from having one entry per neighborhood to one entry per
    # interpolation point
    coeff = coeff[inv]
    y = y[inv]
    centers = centers[inv]
    # evaluate at the interpolation points
    x = x - centers
    phi_coeff = coeff[:, :self.k]
    poly_coeff = coeff[:, self.k:]
    Kxy = self.phi(x[:, None], y, eps=self.eps, diff=diff)[:, 0]
    Px = mvmonos(x, self.order, diff=diff)
    out = (Kxy*phi_coeff).sum(axis=1) + (Px*poly_coeff).sum(axis=1)
    return out
