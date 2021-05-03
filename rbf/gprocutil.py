import logging
import warnings
from functools import wraps

import numpy as np

from rbf.gproc import _as_covariance, GaussianProcess
from rbf.utils import assert_shape
from rbf.linalg import as_sparse_or_array, PartitionedPosDefSolver
LOGGER = logging.getLogger(__name__)


def differentiator(delta):
    '''
    Decorator that makes a function differentiable. The derivatives of the
    function are approximated by finite differences. The function must take a
    single (N, D) array of positions as input. The returned function takes a
    single (N, D) array of positions and a (D,) array derivative specification.

    Parameters
    ----------
    delta : float
        step size to use for finite differences

    '''
    def _differentiator(fin):
        @wraps(fin)
        def fout(x, diff):
            if not any(diff):
                out = np.asarray(fin(x), dtype=float)
                return out

            else:
                diff_axis = np.argmax(diff)
                x_plus_dx = np.copy(x)
                x_plus_dx[:, diff_axis] += delta
                diff_minus_one = np.copy(diff)
                diff_minus_one[diff_axis] -= 1
                out = ( fout(x_plus_dx, diff_minus_one) -
                        fout(x,         diff_minus_one) ) / delta
                return out

        return fout

    return _differentiator


def covariance_differentiator(delta):
    '''
    Decorator that makes a covariance function differentiable. The derivatives
    of the covariance function are approximated by finite differences. The
    covariance function must take an (N, D) array and an (M, D) array of
    positions as input. The returned function takes an (N, D) array and an (M,
    D) array of positions and two (D,) array derivative specifications.

    Parameters
    ----------
    delta : float
        step size to use for finite differences

    '''
    def _covariance_differentiator(fin):
        @wraps(fin)
        def fout(x1, x2, diff1, diff2):
            if (not any(diff1)) & (not any(diff2)):
                return as_sparse_or_array(fin(x1, x2))

            elif any(diff1):
                diff1_axis = np.argmax(diff1)
                x1_plus_dx = np.copy(x1)
                x1_plus_dx[:, diff1_axis] += delta
                diff1_minus_one = np.copy(diff1)
                diff1_minus_one[diff1_axis] -= 1
                out = ( fout(x1_plus_dx, x2, diff1_minus_one, diff2) -
                        fout(x1,         x2, diff1_minus_one, diff2) ) / delta
                return out

            else:
                # any(diff2) == True
                diff2_axis = np.argmax(diff2)
                x2_plus_dx = np.copy(x2)
                x2_plus_dx[:, diff2_axis] += delta
                diff2_minus_one = np.copy(diff2)
                diff2_minus_one[diff2_axis] -= 1
                out = ( fout(x1, x2_plus_dx, diff1, diff2_minus_one) -
                        fout(x1,         x2, diff1, diff2_minus_one) ) / delta
                return out

        return fout

    return _covariance_differentiator


def outliers(d, dsigma, pcov, pmu=None, pbasis=None, tol=4.0, maxitr=50):
    '''
    Uses a data editing algorithm to identify outliers in `d`. Outliers are
    considered to be the data that are abnormally inconsistent with a
    multivariate normal distribution with mean `pmu`, covariance `pcov`, and
    basis vectors `pbasis`.

    The data editing algorithm first conditions the prior with the
    observations, then it compares each residual (`d` minus the expected value
    of the posterior divided by `dsigma`) to the RMS of residuals. Data with
    residuals greater than `tol` times the RMS are identified as outliers. This
    process is then repeated using the subset of `d` which were not flagged as
    outliers. If no new outliers are detected in an iteration then the
    algorithm stops.

    Parameters
    ----------
    d : (N,) float array
        Observations.

    dsigma : (N,) float array
        One standard deviation uncertainty on the observations.

    pcov : (N, N) array, or (N, N) scipy sparse matrix
        Covariance of the prior at the observation points.

    pmu : (N,) float array, optional
        Mean of the prior at the observation points. Defaults to zeros.

    pbasis : (N, P) float array, optional
        Basis functions of the prior evaluated at the observation points.
        Defaults to an (N, 0) array.

    tol : float, optional
        Outlier tolerance. Smaller values make the algorithm more likely to
        identify outliers. A good value is 4.0 and this should not be set any
        lower than 2.0.

    maxitr : int, optional
        Maximum number of iterations.

    Returns
    -------
    out : (N,) bool array
        Array indicating which data are outliers

    '''
    d = np.asarray(d, dtype=float)
    assert_shape(d, (None,), 'd')
    n = d.shape[0]

    dsigma = np.asarray(dsigma, dtype=float)
    assert_shape(dsigma, (n,), 'dsigma')

    pcov = as_sparse_or_array(pcov, dtype=float)
    assert_shape(pcov, (n, n), 'pcov')

    if pmu is None:
        pmu = np.zeros((n,), dtype=float)
    else:      
        pmu = np.asarray(pmu, dtype=float)
        assert_shape(pmu, (n,), 'pmu')

    if pbasis is None:
        pbasis = np.zeros((n, 0), dtype=float)
    else:
        pbasis = np.asarray(pbasis, dtype=float)
        assert_shape(pbasis, (n, None), 'pbasis')

    # total number of outlier detection iterations completed thus far
    itr = 0
    # boolean array indicating outliers
    inliers = np.ones(n, dtype=bool)
    while True:
        LOGGER.debug(
            'Starting iteration %d of outlier detection' % (itr+1)
            )
        # remove rows and cols corresponding to the outliers
        pcov_i = pcov[:, inliers][inliers, :]
        pmu_i = pmu[inliers]
        pbasis_i = pbasis[inliers]
        d_i = d[inliers]
        dsigma_i = dsigma[inliers]
        # add data covariance to prior covariance. If an array is added to a
        # sparse matrix then the output is a matrix. as_sparse_or_array coerces
        # it back to an array
        pcov_i = as_sparse_or_array(pcov_i + _as_covariance(dsigma_i))
        solver = PartitionedPosDefSolver(pcov_i, pbasis_i)
        vec1, vec2 = solver.solve(d_i - pmu_i)

        # dereference everything that we no longer need
        del pcov_i, pmu_i, pbasis_i, d_i, dsigma_i, solver

        fit = pmu + pcov[:, inliers].dot(vec1) + pbasis.dot(vec2)
        # find new outliers based on the misfit
        res = np.abs(fit - d)/dsigma
        rms = np.sqrt(np.mean(res[inliers]**2))
        new_inliers = res < tol*rms
        if np.all(inliers == new_inliers):
            break
        else:
            inliers = new_inliers
            itr += 1
            if itr == maxitr:
                warnings.warn('Reached the maximum number of iterations')
                break

    LOGGER.debug(
        'Detected %s outliers out of %s observations' % 
        (inliers.size - inliers.sum(), inliers.size)
        )

    outliers = ~inliers
    return outliers


def gpgibbs(ls, sigma, delta=1e-4):
    '''
    Returns a `GaussianProcess` with zero mean and a Gibbs covariance function.
    The Gibbs kernel has a spatially varying lengthscale.

    Parameters
    ----------
    ls: function
        Function that takes an (N, D) array of positions and returns an (N, D)
        array indicating the lengthscale along each dimension at those
        positions.

    sigma: float
        Standard deviation of the Gaussian process.

    delta: float, optional
        Finite difference spacing to use when calculating the derivative of the
        `GaussianProcess`. An analytical solution for the derivative is not
        available because the derivative of the `ls` function is unknown.

    '''
    @covariance_differentiator(delta)
    def covariance(x1, x2):
        '''
        covariance function for the Gibbs Gaussian process.
        '''
        dim = x1.shape[1]
        lsx1 = ls(x1)
        lsx2 = ls(x2)

        # sanitize the output for `ls`
        lsx1 = np.asarray(lsx1, dtype=float)
        lsx2 = np.asarray(lsx2, dtype=float)
        assert_shape(lsx1, x1.shape, 'ls(x1)')
        assert_shape(lsx2, x2.shape, 'ls(x2)')

        coeff = np.ones((x1.shape[0], x2.shape[0]))
        exponent = np.zeros((x1.shape[0], x2.shape[0]))

        for i in range(dim):
            a = 2 * lsx1[:, None, i] * lsx2[None, :, i]
            b = lsx1[:, None, i]**2 + lsx2[None, :, i]**2
            coeff *= np.sqrt( a / b )

        for i in range(dim):
            a = ( x1[:, None, i] - x2[None, :, i] )**2
            b = lsx1[:, None, i]**2 + lsx2[None, :, i]**2
            exponent -= ( a / b )

        out = sigma**2*coeff*np.exp(exponent)
        return out

    out = GaussianProcess(covariance=covariance, wrap=False)
    return out
