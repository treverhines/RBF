'''
This module provides a class for RBF interpolation, `RBFInterpolant`.

RBF Interpolation
-----------------
An RBF interpolant fits scalar valued observations
:math:`\mathbf{d}=[d_1,...,d_N]^T` made at the distinct scattered locations
:math:`y_1,...,y_N`. The RBF interpolant is parameterized as

.. math::
    f(x) = \sum_{i=1}^N a_i \phi(||x - y_i||_2) +
           \sum_{j=1}^M b_j p_j(x)

where :math:`\phi` is an RBF and :math:`p_1(x),...,p_M(x)` are monomials that
span the space of polynomials with a specified degree. The coefficients
:math:`\mathbf{a}=[a_1,...,a_N]^T` and :math:`\mathbf{b}=[b_1,...,b_M]^T`
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
from scipy.optimize import minimize

from rbf.linalg import PartitionedSolver, PosDefSolver
from rbf.poly import monomial_count, mvmonos
from rbf.basis import get_rbf
from rbf.utils import assert_shape, KDTree


logger = logging.getLogger(__name__)


def _gml(d, K, P):
    '''
    Generalized Maximum Likelihood (GML) score for RBF interpolation. The
    implementation follows section 4.8 of [1].

    Parameters
    ----------
    d : (N, D) ndarray
        Data values.

    K : (N, N) ndarray
        RBF matrix with smoothing added to diagonals.

    P : (N, M) ndarray
        Polynomial matrix.

    Returns
    -------
    float

    References
    ----------
    [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    '''
    n, m = P.shape
    Q, _ = np.linalg.qr(P, mode='complete')
    Q2 = Q[:, m:]
    K_proj = Q2.T.dot(K).dot(Q2)
    d_proj = Q2.T.dot(d)
    try:
        # Even though `K` may not be positive definite, it should be positive
        # definite when projected to the space orthogonal to the monomials
        # (assuming the degree of the monomials is sufficiently high).
        factor = PosDefSolver(K_proj, factor_inplace=True)
    except np.linalg.LinAlgError:
        return np.nan

    # compute and sum the Mahalanobis distance for each component of `d_proj`
    # using `K_proj` as the covariance matrix. This is in the numerator of the
    # GML expression.
    weighted_d_proj = factor.solve_L(d_proj)
    dist = np.linalg.norm(weighted_d_proj)**2
    # compute the determinant of `K_proj` using the diagonals of its Cholesky
    # decomposition. This is in the denominator of the GML expression.
    logdet = factor.log_det()
    out = dist*np.exp(logdet/(n - m))
    return out


def _loocv(d, K, P):
    '''
    Leave-one-out cross validation. The implementation follows eq. 17.1 of [1].

    Parameters
    ----------
    d : (N, D) ndarray
        Data values.

    K : (N, N) ndarray
        RBF matrix with smoothing added to diagonals.

    P : (N, M) ndarray
        Polynomial matrix.

    Returns
    -------
    float

    References
    ----------
    [1] Fasshauer, G., 2007. Meshfree Approximation Methods with MATLAB. World
    Scientific Publishing Co.

    '''
    n, s = d.shape
    m = P.shape[1]
    A = np.zeros((n + m, n + m), dtype=float)
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.nan

    soln = Ainv[:, :n].dot(d)
    errors = soln[:n] / np.diag(Ainv[:n, :n])[:, None]
    out = np.linalg.norm(errors)
    return out


def _sanitize_arguments(y, d, sigma, phi, eps, order, neighbors):
    '''
    Sanitize the arguments for `RBFInterpolant` and `_objective`.
    '''
    y = np.asarray(y, dtype=float)
    assert_shape(y, (None, None), 'y')
    n, dim = y.shape

    if np.iscomplexobj(d):
        d_dtype = complex
    else:
        d_dtype = float

    d = np.asarray(d, dtype=d_dtype)
    assert_shape(d, (n, ...), 'd')

    if np.isscalar(sigma):
        sigma = np.full(n, sigma, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
        assert_shape(sigma, (n,), 'sigma')

    phi = get_rbf(phi)

    if not np.isscalar(eps):
        raise ValueError('`eps` must be a scalar.')

    # if cpd_order is unknown, dont warn about polynomial degrees
    min_order = phi.cpd_order - 1 if phi.cpd_order is not None else -1
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

    r = monomial_count(order, dim)
    if neighbors is None:
        nobs = n
    else:
        # make sure the number of neighbors does not exceed the number of
        # observations.
        neighbors = int(min(neighbors, n))
        nobs = neighbors

    if r > nobs:
        raise ValueError(
            'At least %d data points are required when `order` is %d and '
            'the number of dimensions is %d.' % (r, order, dim)
            )

    return y, d, sigma, phi, eps, order, neighbors


def _objective(method, y, d, sigma, phi, eps, order):
    '''
    Returns either the LOOCV or GML score for the RBF interpolant.
    '''
    y, d, sigma, phi, eps, order, _ = _sanitize_arguments(
        y, d, sigma, phi, eps, order, None
        )

    d = d.reshape((d.shape[0], -1))
    d = d.view(float)

    K = phi(y, y, eps=eps)
    if sp.issparse(K):
        raise NotImplementedError(
            'Sparse RBFs are not supported for %s.' % method
            )

    K[range(y.shape[0]), range(y.shape[0])] += sigma**2

    maxs = y.max(axis=0)
    mins = y.min(axis=0)
    shift = (maxs + mins)/2
    scale = (maxs - mins)/2
    scale[scale == 0.0] = 1.0
    P = mvmonos((y - shift)/scale, order)

    if method == 'LOOCV':
        return _loocv(d, K, P)
    elif method == 'GML':
        return _gml(d, K, P)
    else:
        raise ValueError('method must be "LOOCV" or "GML".')


def _optimal_sigma_and_eps(y, d, sigma, phi, eps, order):
    '''
    Optimizes `sigma` and/or `eps` if they have the value "auto". They are
    optimized with respect to the LOOCV score.
    '''
    eps_is_auto = isinstance(eps, str) and (eps == 'auto')
    sigma_is_auto = isinstance(sigma, str) and (sigma == 'auto')
    if eps_is_auto or sigma_is_auto:
        if eps_is_auto:
            # Use the average distance to the nearest neighbor to make an
            # initial guess for the shape parameter. This is a heuristic. `y`
            # and `phi` are required but have not yet been sanitized.
            y = np.asarray(y, dtype=float)
            assert_shape(y, (None, None), 'y')

            phi = get_rbf(phi)

            dist = np.mean(KDTree(y).query(y, 2)[0][:, 1])
            if phi.eps_is_divisor:
                eps_init = 2*dist
            elif phi.eps_is_factor:
                eps_init = 1/(2*dist)
            else:
                eps_init = 1.0

        if sigma_is_auto:
            sigma_init = 1.0

        if eps_is_auto and sigma_is_auto:
            result = minimize(
                lambda p: _objective(
                    "LOOCV", y, d, np.exp(p[0]), phi, np.exp(p[1]), order
                    ),
                [np.log(sigma_init), np.log(eps_init)],
                method='Nelder-Mead'
                )

            sigma, eps = np.exp(result.x)
            if not result.success:
                logger.warning('Failed to optimize `sigma` and `eps`.')

            logger.info(
                'sigma: %s, eps: %s, LOOCV: %s' % (sigma, eps, result.fun)
                )

        elif eps_is_auto:
            result = minimize(
                lambda p: _objective(
                    "LOOCV", y, d, sigma, phi, np.exp(p[0]), order
                    ),
                [np.log(eps_init)],
                method='Nelder-Mead'
                )

            eps, = np.exp(result.x)
            if not result.success:
                logger.warning('Failed to optimize `eps`.')

            logger.info('eps: %s, LOOCV: %s' % (eps, result.fun))

        else:
            result = minimize(
                lambda p: _objective(
                    "LOOCV", y, d, np.exp(p[0]), phi, eps, order
                    ),
                [np.log(sigma_init)],
                method='Nelder-Mead'
                )

            sigma, = np.exp(result.x)
            if not result.success:
                logger.warning('Failed to optimize `sigma`.')

            logger.info('sigma: %s, LOOCV: %s' % (sigma, result.fun))

    return sigma, eps


def _build_and_solve_systems(y, d, sigma, phi, eps, order, check_cond):
    '''
    Build the RBF interpolation system of equations and solve for the
    coefficients.

    Parameters
    ----------
    y : (..., N, D) array
        Observation points.

    d : (..., N, S) array
        Observed values at `y`.

    sigma : (..., N) array
        Smoothing parameters for each observation point.

    phi : RBF instance

    eps : float

    order : int

    Returns
    -------
    (..., N, S) array
        solved RBF coefficients.

    (..., R, S) array
        solved polynomial coefficients.

    (..., D) array
        Domain shifts used for the polynomial terms.

    (..., D) array
        Scale factors used for the polynomial terms.

    '''
    bcast = y.shape[:-2]
    n, dim = y.shape[-2:]
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
        # Improve the condition number of the system by rescaling the
        # polynomial matrix, which currently should have values in [-1, 1], to
        # have the same ptp as the kernel matrix.
        Py_scale = (Kyy.max() - Kyy.min())/2

    else:
        Kyy[..., range(n), range(n)] += sigma**2
        Py_scale = Kyy.reshape(bcast + (-1,)).ptp(axis=-1)/2

    Py_scale = np.where(Py_scale==0.0, 1.0, Py_scale)
    Py *= Py_scale[..., None, None]
    if len(bcast) == 0:
        # PartitionedSolver supports solving sparse systems, so use it if
        # possible.
        solver = PartitionedSolver(Kyy, Py, check_cond=check_cond)
        phi_coeff, poly_coeff = solver.solve(d)
    else:
        r = Py.shape[-1]
        LHS = np.zeros(bcast + (n + r, n + r), dtype=float)
        LHS[..., :n, :n] = Kyy
        LHS[..., :n, n:] = Py
        LHS[..., n:, :n] = Py.swapaxes(-2, -1)
        rhs = np.zeros(bcast + (n + r, s), dtype=float)
        rhs[..., :n, :] = d
        coeff = np.linalg.solve(LHS, rhs)
        phi_coeff = coeff[..., :n, :]
        poly_coeff = coeff[..., n:, :]

    poly_coeff *= Py_scale[..., None, None]
    return phi_coeff, poly_coeff, shift, scale


class RBFInterpolant(object):
    '''
    Radial basis function interpolant for scattered data.

    Parameters
    ----------
    y : (N, D) float array
        Observation points.

    d : (N, ...) float or complex array
        Observed values at `y`.

    sigma : float, (N,) array, or "auto", optional
        Smoothing parameter. Setting this to 0 causes the interpolant to
        perfectly fit the data. Increasing the smoothing parameter degrades the
        fit while improving the smoothness of the interpolant. If this is a
        vector, it should be proportional to the one standard deviation
        uncertainties for the observations. This defaults to 0.0.

        If this is set to "auto", then the smoothing parameter will be chosen
        to minimize the leave-one-out cross validation score.

    eps : float or "auto", optional
        Shape parameter. Defaults to 1.0.

        If this is set to "auto", then the shape parameter will be chosen to
        minimize the leave-one-out cross validation score. Polyharmonic splines
        (those starting with "phs") are scale-invariant, and there is no
        purpose to optimizing the shape parameter.

    phi : rbf.basis.RBF instance or str, optional
        The type of RBF. This can be an `rbf.basis.RBF` instance or the RBF
        name as a string. See `rbf.basis` for the available options. Defaults
        to "phs3". If the data is two-dimensional, then setting this to "phs2"
        is equivalent to creating a thin-plate spline.

    order : int, optional
        Order of the added polynomial terms. Set this to -1 for no added
        polynomial terms. If `phi` is a conditionally positive definite RBF of
        order `q`, then this value should be at least `q - 1`. Defaults to the
        `max(q - 1, 0)`.

    neighbors : int, optional
        If given, create an interpolant at each evaluation point using this
        many nearest observations. Defaults to using all the observations.

    check_cond : bool, optional
        Whether to check the condition number of the system being solved. A
        warning is raised if it is ill-conditioned.

    Notes
    -----
    If `sigma` or `eps` are set to "auto", they are optimized with a single run
    of `scipy.optimize.minimize`, using the `loocv` method as the objective
    function. The initial guesses for `sigma` is 1.0, and the initial guess for
    `eps` is a function of the average nearest neighbor distance in `y`. The
    optimization is not guaranteed to succeed. The methods `loocv` or `gml`
    (generalized maximum likelihood) are available for the user to perform the
    optimization on their own.

    References
    ----------
    [1] Fasshauer, G., Meshfree Approximation Methods with Matlab, World
    Scientific Publishing Co, 2007.

    '''
    @staticmethod
    def gml(y, d, sigma=0.0, phi='phs3', eps=1.0, order=None):
        '''
        Generalized maximum likelihood (GML). Optimal values for `sigma` and
        `eps` can be found by minimizing the value returned by this method.

        References
        ----------
        [1] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

        '''
        return _objective("GML", y, d, sigma, phi, eps, order)

    @staticmethod
    def loocv(y, d, sigma=0.0, phi='phs3', eps=1.0, order=None):
        '''
        Leave-one-out cross validation (LOOCV). Optimal values for `sigma` and
        `eps` can be found by minimizing the value returned by this method.

        References
        ----------
        [1] Fasshauer, G., 2007. Meshfree Approximation Methods with MATLAB.
        World Scientific Publishing Co.

        '''
        return _objective("LOOCV", y, d, sigma, phi, eps, order)

    def __init__(self, y, d,
                 sigma=0.0,
                 phi='phs3',
                 eps=1.0,
                 order=None,
                 neighbors=None,
                 check_cond=True):
        sigma, eps = _optimal_sigma_and_eps(y, d, sigma, phi, eps, order)

        y, d, sigma, phi, eps, order, neighbors = _sanitize_arguments(
            y, d, sigma, phi, eps, order, neighbors
            )

        self.d_dtype = d.dtype
        self.d_shape = d.shape[1:]
        d = d.reshape((d.shape[0], -1))
        # If d is complex, turn it into a float with twice as many columns
        d = d.view(float)

        if neighbors is None:
            phi_coeff, poly_coeff, shift, scale = _build_and_solve_systems(
                y, d, sigma, phi, eps, order, check_cond
                )

            self.phi_coeff = phi_coeff
            self.poly_coeff = poly_coeff
            self.shift = shift
            self.scale = scale

        else:
            self.tree = KDTree(y)

        self.y = y
        self.d = d
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
        x : (N, D) float array
            Evaluation points.

        diff : (D,) int array, optional
            Derivative order for each spatial dimension.

        chunk_size : int, optional
            Break `x` into chunks with this size and evaluate the interpolant
            for each chunk. This is use to balance performance and memory
            usage, and it does not affect the output.

        Returns
        -------
        (N, ...) float or complex array

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.y.shape[1]), 'x')
        n, dim = x.shape

        if diff is None:
            diff = np.zeros((dim,), dtype=int)
        else:
            diff = np.asarray(diff, dtype=int)
            assert_shape(diff, (dim,), 'diff')

        if (chunk_size is not None) and (n > chunk_size):
            out = np.zeros((n,) + self.d_shape, dtype=self.d_dtype)
            for start in range(0, n, chunk_size):
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
                y, d, sigma, self.phi, self.eps, self.order, False
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
        out = out.reshape((n,) + self.d_shape)
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
