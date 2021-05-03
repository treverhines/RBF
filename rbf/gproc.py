'''
Eventual replacement to gauss.py
'''
from functools import wraps

import numpy as np
import scipy.sparse as sp

import rbf.poly
import rbf.basis
import rbf.linalg
from rbf.utils import assert_shape, get_arg_count
from rbf.linalg import (as_array, as_sparse_or_array,
                        is_positive_definite, PosDefSolver,
                        PartitionedPosDefSolver)


# If a matrix can be made sparse (e.g., a diagonal matrix or matrix of zeros),
# then make it sparse if its size exceeds this value. Otherwise, make it dense
# to avoid the sparse matrix overhead.
_SPARSE_LIMIT = 64**2


def _as_covariance(sigma):
    '''
    Return `sigma` as a covariance matrix. where `sigma` can be a 1-D
    array-like or a 2-D square array-like. If `sigma` is 1-D, it is treated as
    a vector of standard deviations.
    '''
    ndim = np.ndim(sigma)
    if ndim == 1:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.size**2 <= _SPARSE_LIMIT:
            cov = np.diag(sigma**2)
        else:
            cov = sp.diags(sigma**2).tocsc()

    elif ndim == 2:
        cov = as_sparse_or_array(sigma, dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError('`sigma` must be square if it is 2-dimensional')

    else:
        raise ValueError('`sigma` must be a 1 or 2-dimensional array')

    return cov


def zero_mean(x, diff):
    '''Mean function that returns zeros.'''
    return np.zeros((x.shape[0],), dtype=float)


def zero_variance(x, diff):
    '''Variance function that returns zeros.'''
    return np.zeros((x.shape[0],), dtype=float)


def zero_covariance(x1, x2, diff1, diff2):
    '''Covariance function that returns zeros (either sparse or dense).'''
    if (x1.shape[0]*x2.shape[0]) <= _SPARSE_LIMIT:
        return np.zeros((x1.shape[0], x2.shape[0]), dtype=float)
    else:
        return sp.csc_matrix((x1.shape[0], x2.shape[0]), dtype=float)


def empty_basis(x, diff):
    '''Empty set of basis functions.'''
    return np.zeros((x.shape[0], 0), dtype=float)


def naive_variance_constructor(covariance):
    '''Converts a covariance function to a variance function.'''
    def naive_variance(x, diff):
        cov = covariance(x, x, diff, diff)
        # cov may be a CSC sparse matrix or an array. Either way, it has a
        # diagonal method
        out = cov.diagonal()
        return out

    return naive_variance


def sample(mu, cov, use_cholesky=False, count=None):
    '''
    Draws a random sample from the multivariate normal distribution.

    Parameters
    ----------
    mu : (N,) array

    cov : (N, N) array or sparse matrix

    use_cholesky : bool, optional

    count : int, optional

    Returns
    -------
    (N,) or (count, N) array

    '''
    mu = np.asarray(mu)
    assert_shape(mu, (None,), 'mu')
    n = mu.shape[0]

    cov = as_sparse_or_array(cov)
    assert_shape(cov, (n, n), 'cov')

    if use_cholesky:
        # draw a sample using a cholesky decomposition. This assumes that `cov`
        # is numerically positive definite (i.e. no small negative eigenvalues
        # from rounding error).
        L = PosDefSolver(cov).L()
        if count is None:
            w = np.random.normal(0.0, 1.0, n)
            u = mu + L.dot(w)
        else:
            w = np.random.normal(0.0, 1.0, (n, count))
            u = (mu[:, None] + L.dot(w)).T

    else:
        # otherwise use an eigenvalue decomposition, ignoring negative
        # eigenvalues. If `cov` is sparse then begrudgingly make it dense.
        cov = as_array(cov)
        vals, vecs = np.linalg.eigh(cov)
        keep = (vals > 0.0)
        vals = np.sqrt(vals[keep])
        vecs = vecs[:, keep]
        if count is None:
            w = np.random.normal(0.0, vals)
            u = mu + vecs.dot(w)
        else:
            w = np.random.normal(0.0, vals[:, None].repeat(count, axis=1))
            u = (mu[:, None] + vecs.dot(w)).T

    return u


def likelihood(d, mu, cov, basis=None):
    '''
    Returns the log likelihood of observing `d` from a multivariate normal
    distribution with mean `mu` and covariance `cov`.

    When `basis` is specified, the restricted likelihood is returned. The
    restricted likelihood is the probability of observing `R.dot(d)` from a
    normally distributed random vector with mean `R.dot(mu)` and covariance
    `R.dot(sigma).dot(R.T)`, where `R` is a matrix with rows that are
    orthogonal to the columns of `basis`. In other words, if `basis` is
    specified then the component of `d` which lies along the columns of `basis`
    will be ignored. The restricted likelihood was first described by [1] and
    it is covered in more general reference books such as [2]. Both [1] and [2]
    are good sources for additional information.

    Parameters
    ----------
    d : (N,) array

    mu : (N,) array

    cov : (N, N) array or sparse matrix

    basis : (N, M) array, optional

    Returns
    -------
    float

    References
    ----------
    [1] Harville D. (1974). Bayesian Inference of Variance Components Using
    Only Error Contrasts. Biometrica.

    [2] Cressie N. (1993). Statistics for Spatial Data. John Wiley & Sons.

    '''
    d = np.asarray(d, dtype=float)
    assert_shape(d, (None,), 'd')
    n = d.shape[0]

    mu = np.asarray(mu, dtype=float)
    assert_shape(mu, (n,), 'mu')

    cov = as_sparse_or_array(cov)
    assert_shape(cov, (n, n), 'cov')

    if basis is None:
        basis = np.zeros((n, 0), dtype=float)
    else:
        basis = np.asarray(basis, dtype=float)
        assert_shape(basis, (n, None), 'basis')

    m = basis.shape[1]

    A = PosDefSolver(cov)
    B = A.solve_L(basis)
    C = PosDefSolver(B.T.dot(B))
    D = PosDefSolver(basis.T.dot(basis))

    a = A.solve_L(d - mu)
    b = C.solve_L(B.T.dot(a))

    out = 0.5*(D.log_det() -
               A.log_det() -
               C.log_det() -
               a.T.dot(a) +
               b.T.dot(b) -
               (n-m)*np.log(2*np.pi))
    return out


def _func_wrapper(fin, ftype):
    '''
    Wraps a mean, variance, covariance, or basis function to ensure that it
    takes the correct number of positional arguments and returns an array with
    the correct shape.
    '''
    if fin is None:
        return None

    arg_count = get_arg_count(fin)
    if (ftype == 'mean') | (ftype == 'variance'):
        @wraps(fin)
        def fout(x, diff):
            if arg_count == 1:
                # `fin` only takes one argument and is assumed to not be
                # differentiable
                if any(diff):
                    raise ValueError(
                        'The %s function is not differentiable.' % ftype
                        )

                out = fin(x)
            else:
                # otherwise it is assumed that `fin` takes two arguments
                out = fin(x, diff)

            out = np.asarray(out, dtype=float)
            assert_shape(out, (x.shape[0],), '%s output' % ftype)
            return out

    elif ftype == 'basis':
        @wraps(fin)
        def fout(x, diff):
            if arg_count == 1:
                # `fin` only takes one argument and is assumed to not be
                # differentiable
                if any(diff):
                    raise ValueError(
                        'The basis function is not differentiable.'
                        )

                out = fin(x)
            else:
                # otherwise it is assumed that `fin` takes two arguments
                out = fin(x, diff)

            out = np.asarray(out, dtype=float)
            assert_shape(out, (x.shape[0], None), 'basis output')
            return out

    elif ftype == 'covariance':
        @wraps(fin)
        def fout(x1, x2, diff1, diff2):
            if arg_count == 2:
                # `fin` only takes two argument and is assumed to not be
                # differentiable
                if any(diff1) | any(diff2):
                    raise ValueError(
                        'The covariance function is not differentiable.'
                        )

                out = fin(x1, x2)
            else:
                # otherwise it is assumed that `fin` takes four arguments
                out = fin(x1, x2, diff1, diff2)

            out = as_sparse_or_array(out, dtype=float)
            assert_shape(out, (x1.shape[0], x2.shape[0]), 'covariance output')
            return out

    else:
        raise ValueError

    return fout


def _add(gp1, gp2):
    '''
    Returns a `GaussianProcess` which is the sum of two `GaussianProcess`.
    '''
    if gp2.dim is None:
        dim = gp1.dim
    elif gp1.dim is None:
        dim = gp2.dim
    elif gp1.dim != gp2.dim:
        raise ValueError(
            'The `GaussianProcess` instances have an inconsistent number of '
            'dimensions.'
            )
    else:
        dim = gp1.dim

    if gp2._mean is None:
        added_mean = gp1._mean
    elif gp1._mean is None:
        added_mean = gp2._mean
    else:
        def added_mean(x, diff):
            out = gp1._mean(x, diff) + gp2._mean(x, diff)
            return out

    if gp2._variance is None:
        added_variance = gp1._variance
    elif gp1._variance is None:
        added_variance = gp2._variance
    else:
        def added_variance(x, diff):
            out = gp1._variance(x, diff) + gp2._variance(x, diff)
            return out

    if gp2._covariance is None:
        added_covariance = gp1._covariance
    elif gp1._covariance is None:
        added_covariance = gp2._covariance
    else:
        def added_covariance(x1, x2, diff1, diff2):
            out = as_sparse_or_array(
                gp1._covariance(x1, x2, diff1, diff2) +
                gp2._covariance(x1, x2, diff1, diff2)
                )
            return out

    if gp2._basis is None:
        added_basis = gp1._basis
    elif gp1._basis is None:
        added_basis = gp2._basis
    else:
        def added_basis(x, diff):
            out = np.hstack(
                (gp1._basis(x, diff), gp2._basis(x, diff))
                )
            return out

    out = GaussianProcess(
        mean=added_mean,
        covariance=added_covariance,
        variance=added_variance,
        basis=added_basis,
        dim=dim,
        wrap=False
        )
    return out


def _scale(gp, c):
    '''
    Returns a scaled `GaussianProcess`.
    '''
    if gp._mean is None:
        scaled_mean = None
    else:
        def scaled_mean(x, diff):
            out = c*gp._mean(x, diff)
            return out

    if gp._variance is None:
        scaled_variance = None
    else:
        def scaled_variance(x, diff):
            out = c**2*gp._variance(x, diff)
            return out

    if gp._covariance is None:
        scaled_covariance = None
    else:
        def scaled_covariance(x1, x2, diff1, diff2):
            out = c**2*gp._covariance(x1, x2, diff1, diff2)
            return out

    out = GaussianProcess(
        mean=scaled_mean,
        covariance=scaled_covariance,
        basis=gp._basis,
        variance=scaled_variance,
        dim=gp.dim,
        wrap=False
        )
    return out


def _differentiate(gp, d):
    '''
    Differentiates a `GaussianProcess`.
    '''
    if gp._mean is None:
        differentiated_mean = None
    else:
        def differentiated_mean(x, diff):
            out = gp._mean(x, diff + d)
            return out

    if gp._variance is None:
        differentiated_variance = None
    else:
        def differentiated_variance(x, diff):
            out = gp._variance(x, diff + d)
            return out

    if gp._covariance is None:
        differentiated_covariance = None
    else:
        def differentiated_covariance(x1, x2, diff1, diff2):
            out = gp._covariance(x1, x2, diff1 + d, diff2 + d)
            return out

    if gp._basis is None:
        differentiated_basis = None
    else:
        def differentiated_basis(x, diff):
            out = gp._basis(x, diff + d)
            return out

    out = GaussianProcess(
        mean=differentiated_mean,
        covariance=differentiated_covariance,
        basis=differentiated_basis,
        variance=differentiated_variance,
        dim=d.size,
        wrap=False
        )
    return out


def _condition(gp, y, d, dcov, dbasis, ddiff, build_inverse):
    '''
    Returns a conditioned `GaussianProcess`.
    '''
    if gp._mean is None:
        prior_mean = zero_mean
    else:
        prior_mean = gp._mean

    if gp._covariance is None:
        # if _covariance is None then _variance is also None
        prior_covariance = zero_covariance
        prior_variance = zero_variance
    else:
        prior_covariance = gp._covariance
        if gp._variance is None:
            prior_variance = naive_variance_constructor(prior_covariance)
        else:
            prior_variance = gp._variance

    if gp._basis is None:
        prior_basis = empty_basis
    else:
        prior_basis = gp._basis

    # covariance of the observation points
    cov = dcov + prior_covariance(y, y, ddiff, ddiff)
    cov = as_sparse_or_array(cov)

    # residual at the observation points
    res = d - prior_mean(y, ddiff)

    # basis functions at the observation points
    basis = prior_basis(y, ddiff)
    if dbasis.shape[1] != 0:
        basis = np.hstack((basis, dbasis))

    solver = PartitionedPosDefSolver(
        cov, basis, build_inverse=build_inverse
        )

    # precompute these vectors which are used for `posterior_mean`
    vec1, vec2 = solver.solve(res)

    del res, cov, basis

    def posterior_mean(x, diff):
        mu_x = prior_mean(x, diff)
        cov_xy = prior_covariance(x, y, diff, ddiff)
        basis_x = prior_basis(x, diff)
        if dbasis.shape[1] != 0:
            pad = np.zeros((x.shape[0], dbasis.shape[1]), dtype=float)
            basis_x = np.hstack((basis_x, pad))

        out = mu_x + cov_xy.dot(vec1) + basis_x.dot(vec2)
        return out

    def posterior_covariance(x1, x2, diff1, diff2):
        cov_x1x2 = prior_covariance(x1, x2, diff1, diff2)
        cov_x1y = prior_covariance(x1, y, diff1, ddiff)
        cov_x2y = prior_covariance(x2, y, diff2, ddiff)
        basis_x1 = prior_basis(x1, diff1)
        basis_x2 = prior_basis(x2, diff2)
        if dbasis.shape[1] != 0:
            pad = np.zeros((x1.shape[0], dbasis.shape[1]), dtype=float)
            basis_x1 = np.hstack((basis_x1, pad))

            pad = np.zeros((x2.shape[0], dbasis.shape[1]), dtype=float)
            basis_x2 = np.hstack((basis_x2, pad))

        mat1, mat2 = solver.solve(cov_x2y.T, basis_x2.T)
        out = cov_x1x2 - cov_x1y.dot(mat1) - basis_x1.dot(mat2)
        # `out` may either be a matrix or array depending on whether cov_x1x2
        # is sparse or dense. Make the output consistent by converting to array
        out = np.asarray(out)
        return out

    def posterior_variance(x, diff):
        var_x = prior_variance(x, diff)
        cov_xy = prior_covariance(x, y, diff, ddiff)
        basis_x = prior_basis(x, diff)
        if dbasis.shape[1] != 0:
            pad = np.zeros((x.shape[0], dbasis.shape[1]), dtype=float)
            basis_x = np.hstack((basis_x, pad))

        mat1, mat2 = solver.solve(cov_xy.T, basis_x.T)
        # Efficiently get the diagonals of C_xy.dot(mat1) and p_x.dot(mat2)
        if sp.issparse(cov_xy):
            diag1 = cov_xy.multiply(mat1.T).sum(axis=1).A[:, 0]
        else:
            diag1 = np.einsum('ij, ji -> i', cov_xy, mat1)

        diag2 = np.einsum('ij, ji -> i', basis_x, mat2)
        out = var_x - diag1 - diag2
        return out

    out = GaussianProcess(
        posterior_mean,
        posterior_covariance,
        variance=posterior_variance,
        dim=y.shape[1],
        wrap=False
        )
    return out


class GaussianProcess:
    '''
    A `GaussianProcess` instance represents a stochastic process which is
    defined in terms of a mean function, a covariance function, and
    (optionally) a set of basis functions. This class is used to perform basic
    operations on Gaussian processes which include addition, subtraction,
    scaling, differentiation, sampling, and conditioning.

    Parameters
    ----------
    mean : function, optional
        Function which returns either the mean of the Gaussian process at `x`
        or a specified derivative of the mean at `x`. This has the call
        signature

        `out = mean(x)`

        or

        `out = mean(x, diff)`

        `x` is an (N, D) array of positions. `diff` is a (D,) int array
        derivative specification. `out` must be an (N,) array. If this function
        only takes one argument then it is assumed to not be differentiable and
        the `differentiate` method for the `GaussianProcess` instance will
        return an error.

    covariance : function, optional
        Function which returns either the covariance of the Gaussian process
        between points `x1` and `x2` or the covariance of the specified
        derivatives of the Gaussian process between points `x1` and `x2`. This
        has the call signature

        `out = covariance(x1, x2)`

        or

        `out = covariance(x1, x2, diff1, diff2)`

        `x1` and `x2` are (N, D) and (M, D) arrays of positions, respectively.
        `diff1` and `diff2` are (D,) int array derivative specifications. `out`
        can be an (N, M) array or scipy sparse matrix (csc format would be most
        efficient). If this function only takes two arguments, then it is
        assumed to not be differentiable and the `differentiate` method for the
        `GaussianProcess` instance will return an error.

    basis : function, optional
        Function which returns either the basis functions evaluated at `x` or
        the specified derivative of the basis functions evaluated at `x`. This
        has the call signature

        `out = basis(x)`

        or

        `out = basis(x, diff)`

        `x` is an (N, D) array of positions. `diff` is a (D,) int array
        derivative specification. `out` is an (N, P) array where each column
        corresponds to a basis function. By default, a `GaussianProcess`
        instance contains no basis functions. If this function only takes one
        argument, then it is assumed to not be differentiable and the
        `differentiate` method for the `GaussianProcess` instance will return
        an error.

    variance : function, optional
        A function that returns the variance of the Gaussian process or its
        derivative at `x`. The has the call signature

        `out = variance(x)`

        or

        `out = variance(x, diff)`

        If this function is provided, it should be a more efficient alternative
        to evaluating the covariance matrix at `(x, x)` and then taking the
        diagonals.

    dim : int, optional
        Fixes the spatial dimensions of the `GaussianProcess` instance. An
        error will be raised if method arguments have a conflicting number of
        spatial dimensions.

    Examples
    --------
    Create a `GaussianProcess` describing Brownian motion

    >>> import numpy as np
    >>> from rbf.gauss import GaussianProcess
    >>> def mean(x): return np.zeros(x.shape[0])
    >>> def cov(x1, x2): return np.minimum(x1[:, None, 0], x2[None, :, 0])
    >>> gp = GaussianProcess(mean, cov, dim=1) # Brownian motion is 1D

    '''
    def __init__(self,
                 mean=None,
                 covariance=None,
                 basis=None,
                 variance=None,
                 dim=None,
                 wrap=True):
        if (covariance is None) & (variance is not None):
            raise ValueError(
                '`variance` cannot be specified if `covariance` is not '
                'specified'
                )

        if wrap:
            self._mean = _func_wrapper(mean, 'mean')
            self._covariance = _func_wrapper(covariance, 'covariance')
            self._basis = _func_wrapper(basis, 'basis')
            self._variance = _func_wrapper(variance, 'variance')
        else:
            self._mean = mean
            self._covariance = covariance
            self._basis = basis
            self._variance = variance

        self.dim = dim

    def __repr__(self):
        items = []
        if self._mean is not None:
            items.append('mean=%s' % self._mean.__name__)
        if self._covariance is not None:
            items.append('covariance=%s' % self._covariance.__name__)
        if self._basis is not None:
            items.append('basis=%s' % self._basis.__name__)
        if self.dim is not None:
            items.append('dim=%d' % self.dim)

        out = 'GaussianProcess(%s)' % (', '.join(items))
        return out

    def __add__(self, other):
        '''Equivalent to calling `add`.'''
        return self.add(other)

    def __sub__(self, other):
        '''Equivalent to calling `subtract`.'''
        return self.subtract(other)

    def __mul__(self, c):
        '''Equivalent to calling `scale`.'''
        return self.scale(c)

    def __rmul__(self, c):
        '''Equivalent to calling `scale`.'''
        return self.__mul__(c)

    def __or__(self, args):
        '''Equivalent to calling `condition` with positional arguments.'''
        return self.condition(*args)

    def add(self, other):
        '''Adds two `GaussianProcess` instances.'''
        out = _add(self, other)
        return out

    def subtract(self, other):
        '''Subtracts two `GaussianProcess` instances.'''
        out = _add(self, _scale(other, -1.0))
        return out

    def scale(self, c):
        '''Scales a `GaussianProcess` instance.'''
        c = float(c)
        out = _scale(self, c)
        return out

    def differentiate(self, d):
        '''Returns the derivative of a `GaussianProcess`.'''
        d = np.asarray(d, dtype=int)
        assert_shape(d, (self.dim,), 'd')
        out = _differentiate(self, d)
        return out

    def condition(self, y, d,
                  dsigma=None,
                  dbasis=None,
                  ddiff=None,
                  build_inverse=False):
        '''
        Returns a `GaussianProcess` conditioned on the data.

        Parameters
        ----------
        y : (N, D) float array
            Observation points.

        d : (N,) float array
            Observed values at `y`.

        dsigma : (N,) array, (N, N) array, or (N, N) sparse matrix, optional
            Data uncertainty. If this is an (N,) array then it describes one
            standard deviation of the data error. If this is an (N, N) array
            then it describes the covariances of the data error. If nothing is
            provided then the error is assumed to be zero.

        dbasis : (N, P) array, optional
            Data noise basis vectors. The data noise is assumed to contain some
            unknown linear combination of the columns of `p`.

        ddiff : (D,) int array, optional
            Derivative of the observations. For example, use (1,) if the
            observations are 1-D and should constrain the slope.

        build_inverse : bool, optional
            Whether to construct the inverse matrices rather than just the
            factors.

        Returns
        -------
        out : GaussianProcess

        '''
        y = np.asarray(y, dtype=float)
        assert_shape(y, (None, self.dim), 'y')
        n, dim = y.shape

        d = np.asarray(d, dtype=float)
        assert_shape(d, (n,), 'd')

        if dsigma is None:
            dsigma = np.zeros((n,), dtype=float)

        dcov = _as_covariance(dsigma)
        assert_shape(dcov, (n, n), 'dcov')

        if dbasis is None:
            dbasis = np.zeros((n, 0), dtype=float)
        else:
            dbasis = np.asarray(dbasis, dtype=float)
            assert_shape(dbasis, (n, None), 'dbasis')

        if ddiff is None:
            ddiff = np.zeros(dim, dtype=int)
        else:
            ddiff = np.asarray(ddiff, dtype=int)
            assert_shape(ddiff, (dim,), 'ddiff')

        out = _condition(
            self, y, d, dcov, dbasis, ddiff,
            build_inverse=build_inverse
            )
        return out

    def basis(self, x, diff=None):
        '''
        Returns the basis functions evaluated at `x`.

        Parameters
        ----------
        x : (N, D) array
            Evaluation points.

        diff : (D,) int array
            Derivative specification.

        Returns
        -------
        (N, P) array

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.dim), 'x')
        dim = x.shape[1]

        if diff is None:
            diff = np.zeros(dim, dtype=int)
        else:
            diff = np.asarray(diff, dtype=int)
            assert_shape(diff, (dim,), 'diff')

        if self._basis is None:
            out = empty_basis(x, diff)
        else:
            out = self._basis(x, diff)

        return out

    def mean(self, x, diff=None):
        '''
        Returns the mean of the Gaussian process.

        Parameters
        ----------
        x : (N, D) array
            Evaluation points.

        diff : (D,) int array
            Derivative specification.

        Returns
        -------
        (N,) array

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.dim), 'x')
        dim = x.shape[1]

        if diff is None:
            diff = np.zeros(dim, dtype=int)
        else:
            diff = np.asarray(diff, dtype=int)
            assert_shape(diff, (dim,), 'diff')

        if self._mean is None:
            out = zero_mean(x, diff)
        else:
            out = self._mean(x, diff)

        return out

    def variance(self, x, diff=None):
        '''
        Returns the variance of the Gaussian process.

        Parameters
        ----------
        x : (N, D) array
            Evaluation points.

        diff : (D,) int array
            Derivative specification.

        Returns
        -------
        (N,) array

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.dim), 'x')
        dim = x.shape[1]

        if diff is None:
            diff = np.zeros(dim, dtype=int)
        else:
            diff = np.asarray(diff, dtype=int)
            assert_shape(diff, (dim,), 'diff')

        if self._variance is None:
            if self._covariance is None:
                out = zero_variance(x, diff)
            else:
                out = naive_variance_constructor(self._covariance)(x, diff)
        else:
            out = self._variance(x, diff)

        return out

    def covariance(self, x1, x2, diff1=None, diff2=None):
        '''
        Returns the covariance matrix the Gaussian process.

        Parameters
        ----------
        x1, x2 : (N, D) array
            Evaluation points.

        diff1, diff2 : (D,) int array
            Derivative specifications.

        Returns
        -------
        out : (N, N) array

        '''
        x1 = np.asarray(x1, dtype=float)
        assert_shape(x1, (None, self.dim), 'x1')
        dim = x1.shape[1]

        x2 = np.asarray(x2, dtype=float)
        assert_shape(x2, (None, dim), 'x2')

        if diff1 is None:
            diff1 = np.zeros(dim, dtype=int)
        else:
            diff1 = np.asarray(diff1, dtype=int)
            assert_shape(diff1, (dim,), 'diff1')

        if diff2 is None:
            diff2 = np.zeros(dim, dtype=int)
        else:
            diff2 = np.asarray(diff2, dtype=int)
            assert_shape(diff2, (dim,), 'diff2')

        if self._covariance is None:
            out = zero_covariance(x1, x2, diff1, diff2)
        else:
            out = self._covariance(x1, x2, diff1, diff2)

        # `out` may be sparse or dense, make sure it is dense so that the
        # output is consistent
        out = as_array(out)
        return out

    def __call__(self, x, chunk_size=100):
        '''
        Returns the mean and standard deviation of the Gaussian process.

        Parameters
        ----------
        x : (N, D) array
            Evaluation points.

        chunk_size : int, optional
            Break `x` into chunks with this size for evaluation.

        Returns
        -------
        (N,) array
            Mean at `x`.

        (N,) array
            One standard deviation at `x`.

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.dim), 'x')
        n, dim = x.shape

        diff = np.zeros(dim, dtype=int)

        out_mu = np.empty(n, dtype=float)
        out_sigma = np.empty(n, dtype=float)
        for start in range(0, n, chunk_size):
            stop = start + chunk_size
            out_mu[start:stop] = self.mean(x[start:stop], diff)
            out_sigma[start:stop] = np.sqrt(self.variance(x[start:stop], diff))

        return out_mu, out_sigma

    def likelihood(self, y, d, dsigma=None, dbasis=None):
        '''
        Returns the log likelihood of drawing the observations `d` from this
        `GaussianProcess`. The observations could potentially have noise which
        is described by `dsigma` and `dbasis`. If the Gaussian process contains
        any basis functions or if `dbasis` is specified, then the restricted
        likelihood is returned.

        Parameters
        ----------
        y : (N, D) array
            Observation points.

        d : (N,) array
            Observed values at `y`.

        dsigma : (N,) array, (N, N) array, or (N, N) sparse matrix, optional
            Data uncertainty. If this is an (N,) array then it describes one
            standard deviation of the data error. If this is an (N, N) array
            then it describes the covariances of the data error. If nothing is
            provided then the error is assumed to be zero.

        dbasis : (N, P) float array, optional
            Basis vectors for the noise. The data noise is assumed to contain
            some unknown linear combination of the columns of `p`.

        Returns
        -------
        float
            log likelihood.

        '''
        y = np.asarray(y, dtype=float)
        assert_shape(y, (None, self.dim), 'y')
        n, dim = y.shape

        d = np.asarray(d, dtype=float)
        assert_shape(d, (n,), 'd')

        if dsigma is None:
            dsigma = np.zeros((n,), dtype=float)

        dcov = _as_covariance(dsigma)
        assert_shape(dcov, (n, n), 'dcov')

        if dbasis is None:
            dbasis = np.zeros((n, 0), dtype=float)
        else:
            dbasis = np.asarray(dbasis, dtype=float)
            assert_shape(dbasis, (n, None), 'dbasis')

        mu = self.mean(y)
        cov = dcov + self.covariance(y, y)
        basis = np.hstack((self.basis(y), dbasis))

        out = likelihood(d, mu, cov, basis=basis)
        return out

    def sample(self, x, use_cholesky=False, count=None):
        '''
        Draws a random sample from the Gaussian process.
        '''
        mu = self.mean(x)
        cov = self.covariance(x, x)
        out = sample(mu, cov, use_cholesky=use_cholesky, count=count)
        return out

    def is_positive_definite(self, x):
        '''
        Tests if the covariance function evaluated at `x` is positive definite.
        '''
        cov = self.covariance(x, x)
        out = is_positive_definite(cov)
        return out


def gpiso(phi, epsilon, variance, dim=None):
    '''
    Creates an isotropic `GaussianProcess` instance which has zero mean and a
    covariance function that is described by a radial basis function.

    Parameters
    ----------
    phi : str or RBF instance
        Radial basis function describing the covariance function.

    epsilon : float

    variance : float

    dim : int, optional
        Fixes the spatial dimensions of the domain.

    Returns
    -------
    out : GaussianProcess

    Notes
    -----
    Not all radial basis functions are positive definite, which means that it
    is possible to instantiate an invalid `GaussianProcess`. The method
    `is_positive_definite` provides a necessary but not sufficient test for
    positive definiteness. Examples of predefined `RBF` instances which are
    positive definite include: `rbf.basis.se`, `rbf.basis.ga`, `rbf.basis.exp`,
    `rbf.basis.iq`, `rbf.basis.imq`.

    '''
    phi = rbf.basis.get_rbf(phi)

    def isotropic_covariance(x1, x2, diff1, diff2):
        diff = diff1 + diff2
        coeff = variance*(-1)**sum(diff2)
        out = coeff*phi(x1, x2, eps=epsilon, diff=diff)
        return out

    def isotropic_variance(x, diff):
        coeff = variance*(-1)**sum(diff)
        value = coeff*phi.center_value(eps=epsilon, diff=2*diff)
        out = np.full(x.shape[0], value)
        return out

    out = GaussianProcess(
        covariance=isotropic_covariance,
        variance=isotropic_variance,
        dim=dim,
        wrap=False
        )
    return out


def gppoly(order, dim=None):
    '''
    Returns a `GaussianProcess` consisting of monomial basis functions.

    Parameters
    ----------
    order : int
        Order of the polynomial spanned by the basis functions.

    dim : int, optional
        Fixes the spatial dimensions of the domain.

    Returns
    -------
        out : GaussianProcess

    '''
    def polynomial_basis(x, diff):
        powers = rbf.poly.monomial_powers(order, x.shape[1])
        out = rbf.poly.mvmonos(x, powers, diff)
        return out

    out = GaussianProcess(basis=polynomial_basis, dim=dim, wrap=False)
    return out
