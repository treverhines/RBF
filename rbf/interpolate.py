'''
This module provides a class for RBF interpolation, `RBFInterpolant`.

RBF Interpolation
-----------------
An RBF interpolant fits scalar valued observations
:math:`\mathbf{d}=[d_1,...,d_n]^T` made at the distinct scattered locations
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

where :math:`\mathbf{K}` is a matrix with components
:math:`K_{ij} = \phi(||y_i - y_j||_2)`, :math:`\mathbf{P}` is a matrix with
components :math:`P_{ij}=p_j(y_i)`, and :math:`\sigma` is a smoothing parameter
that controls how well we want to fit the observations. The observations are
fit exactly when :math:`\sigma` is zero.

If the chosen RBF is positive definite (see `rbf.basis`) and :math:`\mathbf{P}`
has full column rank, the solution for :math:`\mathbf{a}` and
:math:`\mathbf{b}` is unique. If the chosen RBF is conditionally positive
definite of order `q` and :math:`\mathbf{P}` has full column rank, the solution
is unique provided that the degree of the monomial terms is at least `q-1` (see
Chapter 7 of [1] or [2]).

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
import scipy.sparse as sp

import rbf.basis
from rbf.linalg import PartitionedSolver
from rbf.poly import monomial_count, mvmonos
from rbf.basis import get_rbf
from rbf.utils import assert_shape, KDTree


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


def _build_and_solve_systems(y, d, sigma, phi, eps, order):
    '''
    Efficiently build and solve `K` different RBF interpolation problems
    with vectorization.

    Parameters
    ----------
    y : (..., P, N) array
        Observation points.

    d : (..., P, S) array
        Observed values at `y`.

    sigma : (..., P) array
        Smoothing parameters for each observation point.

    phi : RBF instance

    eps : float

    order : int

    Returns
    -------
    (..., P, S) array
        solved RBF coefficients.

    (..., R, S) array
        solved polynomial coefficients.

    (..., N) array
        Domain shifts used for the polynomial terms.

    (..., N) array
        Scale factors used for the polynomial terms.

    '''
    bcast = y.shape[:-2]
    p, n = y.shape[-2:]
    s = d.shape[-1]

    maxs = y.max(axis=-2)
    mins = y.min(axis=-2)
    shift = (maxs + mins)/2
    scale = (maxs - mins)/2
    # This happens if there is a single point
    scale[scale == 0.0] = 1.0

    Kyy = phi(y, y, eps=eps)
    Py = mvmonos((y - shift[..., None, :])/scale[..., None, :], order)
    if sp.issparse(Kyy):
        Kyy = sp.csc_matrix(Kyy + sp.diags(sigma**2))
    else:
        Kyy[..., range(p), range(p)] += sigma**2

    if len(bcast) == 0:
        # PartitionedSolver supports solving sparse systems, so use it if
        # possible.
        phi_coeff, poly_coeff = PartitionedSolver(Kyy, Py).solve(d)
    else:
        r = Py.shape[-1]
        Pyt = Py.swapaxes(-2, -1)
        Z = np.zeros(bcast + (r, r), dtype=float)
        z = np.zeros(bcast + (r, s), dtype=float)
        LHS = np.block([[Kyy, Py], [Pyt, Z]])
        rhs = np.concatenate((d, z), axis=-2)
        coeff = np.linalg.solve(LHS, rhs)
        phi_coeff = coeff[..., :p, :]
        poly_coeff = coeff[..., p:, :]

    return phi_coeff, poly_coeff, shift, scale


class RBFInterpolant(object):
    '''
    Radial basis function interpolant for N-dimensional data.

    Parameters
    ----------
    y : (P, N) float array
        Observation points.

    d : (P, ...) float or complex array
        Observed values at `y`.

    sigma : float or (P,) array, optional
        Smoothing parameter. Setting this to 0 causes the interpolant to
        perfectly fit the data. Increasing the smoothing parameter degrades the
        fit while improving the smoothness of the interpolant. If this is a
        vector, it should be proportional to the one standard deviation
        uncertainties for the observations. This defaults to zeros.

    eps : float, optional
        Shape parameter.

    phi : rbf.basis.RBF instance or str, optional
        The type of RBF. This can be an `rbf.basis.RBF` instance or the RBF
        abbreviation as a string. See `rbf.basis` for the available options.

    order : int, optional
        Order of the added polynomial terms. Set this to `-1` for no added
        polynomial terms. If `phi` is a conditionally positive definite RBF of
        order `m`, then this value should be at least `m - 1`.

    neighbors : int, optional
        If given, create an interpolant at each evaluation point using this
        many nearest observations. Defaults to using all the observations.

    References
    ----------
    [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World
    Scientific Publishing Co, 2007.

    '''
    def __init__(self, y, d,
                 sigma=0.0,
                 phi='phs3',
                 eps=1.0,
                 order=None,
                 neighbors=None):
        y = np.asarray(y, dtype=float)
        assert_shape(y, (None, None), 'y')
        p, n = y.shape

        if np.iscomplexobj(d):
            d_dtype = complex
        else:
            d_dtype = float

        d = np.asarray(d, dtype=d_dtype)
        assert_shape(d, (p, ...), 'd')

        d_shape = d.shape[1:]
        d = d.reshape((p, -1))
        # If d is complex, turn it into a float with twice as many columns
        d = d.view(float)

        if np.isscalar(sigma):
            sigma = np.full(p, sigma, dtype=float)
        else:
            sigma = np.asarray(sigma, dtype=float)
            assert_shape(sigma, (p,), 'sigma')

        phi = get_rbf(phi)

        if not np.isscalar(eps):
            raise ValueError('`eps` must be a scalar.')

        # If `phi` is not in `_MIN_ORDER`, then the RBF is either positive
        # definite (no minimum polynomial order) or user-defined (no known
        # minimum polynomial order)
        min_order = _MIN_ORDER.get(phi, -1)
        if order is None:
            order = max(min_order, 0)
        else:
            order = int(order)
            if order < -1:
                raise ValueError('`order` must be at least -1.')

            elif order < min_order:
                logger.warning(
                    'The polynomial order should not be below %d when `phi` '
                    'is %s. The interpolant may not be well-posed.' %
                    (min_order, phi)
                    )

        r = monomial_count(order, n)
        if neighbors is None:
            nobs = p
        else:
            # make sure the number of neighbors does not exceed the number of
            # observations.
            neighbors = int(min(neighbors, p))
            nobs = neighbors

        if r > nobs:
            raise ValueError(
                'At least %d data points are required when `order` is %d and '
                'the number of dimensions is %d.' % (r, order, n)
                )

        if neighbors is None:
            phi_coeff, poly_coeff, shift, scale = _build_and_solve_systems(
                y, d, sigma, phi, eps, order
                )

            self.phi_coeff = phi_coeff
            self.poly_coeff = poly_coeff
            self.shift = shift
            self.scale = scale

        else:
            self.tree = KDTree(y)

        self.y = y
        self.d = d
        self.d_shape = d_shape
        self.d_dtype = d_dtype
        self.sigma = sigma
        self.phi = phi
        self.eps = eps
        self.order = order
        self.neighbors = neighbors


    def __call__(self, x, diff=None, chunk_size=1000):
        '''
        Evaluates the interpolant at `x`.

        Parameters
        ----------
        x : (Q, N) float array
            Evaluation points.

        diff : (N,) int array, optional
            Derivative order for each spatial dimension.

        chunk_size : int, optional
            Break `x` into chunks with this size and evaluate the interpolant
            for each chunk.

        Returns
        -------
        (Q, ...) float or complex array

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.y.shape[1]), 'x')
        q, n = x.shape

        if diff is None:
            diff = np.zeros((n,), dtype=int)
        else:
            diff = np.asarray(diff, dtype=int)
            assert_shape(diff, (n,), 'diff')

        if (chunk_size is not None) and (q > chunk_size):
            out = np.zeros((q,) + self.d_shape, dtype=self.d_dtype)
            for start in range(0, q, chunk_size):
                stop = start + chunk_size
                out[start:stop] = self(x[start:stop], diff, None)

            return out

        if self.neighbors is None:
            Kxy = self.phi(x, self.y, eps=self.eps, diff=diff)
            Px = mvmonos((x - self.shift)/self.scale, self.order, diff=diff)
            Px /= np.prod(self.scale**diff)
            out = Kxy.dot(self.phi_coeff) + Px.dot(self.poly_coeff)

        else:
            # get the indices of the k-nearest observations for each
            # interpolation point
            _, nbr = self.tree.query(x, self.neighbors)
            # multiple interpolation points may have the same neighborhood.
            # Make the neighborhoods unique so that we only compute the
            # interpolation coefficients once for each neighborhood
            nbr = np.sort(nbr, axis=1)
            nbr, inv = np.unique(nbr, return_inverse=True, axis=0)
            # Get the observation data for each neighborhood
            y, d, sigma = self.y[nbr], self.d[nbr], self.sigma[nbr]
            phi_coeff, poly_coeff, shift, scale = _build_and_solve_systems(
                y, d, sigma, self.phi, self.eps, self.order
                )

            # expand the arrays from having one entry per neighborhood to one
            # entry per evaluation point.
            y = y[inv]
            shift = shift[inv]
            scale = scale[inv]
            phi_coeff = phi_coeff[inv]
            poly_coeff = poly_coeff[inv]

            Kxy = self.phi(x[:, None], y, eps=self.eps, diff=diff)[:, 0, :]
            Px = mvmonos((x - shift)/scale, self.order, diff=diff)
            Px /= np.prod(scale**diff, axis=1)[:, None]
            out = (
                np.sum(Kxy[:, :, None]*phi_coeff, axis=1) +
                np.sum(Px[:, :, None]*poly_coeff, axis=1)
                )

        out = out.view(self.d_dtype)
        out = out.reshape((q,) + self.d_shape)
        return out


class KNearestRBFInterpolant(RBFInterpolant):
    '''
    Approximation to `RBFInterpolant` that only uses the k nearest observations
    to each evaluation point.

    Parameters
    ----------
    y : (N, D) array
        Observation points.

    d : (N,) array
        Observed values at `y`.

    sigma : float or (N,) array, optional
        Smoothing parameter. Setting this to 0 causes the interpolant to
        perfectly fit the data. Increasing the smoothing parameter degrades the
        fit while improving the smoothness of the interpolant. If this is a
        vector, it should be proportional to the one standard deviation
        uncertainties for the observations. This defaults to zeros.

    k : int, optional
        Number of neighboring observations to use for each evaluation point.

    eps : float, optional
        Shape parameter.

    phi : rbf.basis.RBF instance or str, optional
        The type of RBF. This can be an `rbf.basis.RBF` instance or the RBF
        abbreviation as a string. See `rbf.basis` for the available options.

    order : int, optional
        Order of the added polynomial terms. Set this to -1 for no added
        polynomial terms. If `phi` is a conditionally positive definite RBF of
        order `m`, then this value should be at least `m - 1`.

    References
    ----------
    [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World
    Scientific Publishing Co, 2007.

    '''
    def __init__(self, *args, k=20, **kwargs):
        logger.warning(
            '`KNearestRBFInterpolant` is deprecated. Use `RBFInterpolant` '
            'with the `neighbors` argument instead.'
            )
        super().__init__(*args, neighbors=k, **kwargs)
