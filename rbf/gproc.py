'''
This module defines the `GaussianProcess` class which is used to perform
Gaussian process regression (GPR) and other operations with Gaussian processes.
GPR is also known as Kriging or Least Squares Collocation.  It is a technique
for constructing a continuous function from discrete observations by
incorporating a stochastic prior model for the underlying function.

There are several existing python packages for Gaussian processes (See
www.gaussianprocess.org for an updated list of packages). This module was
written because existing software lacked support for 1) Gaussian processes with
added basis functions 2) analytical differentiation of Gaussian processes and
3) conditioning a Gaussian process with derivative constraints. Other software
packages have a strong focus on optimizing hyperparameters based on data
likelihood. This module does not include any optimization routines and
hyperparameters are always explicitly specified by the user. However, the
`GaussianProcess` class contains the `log_likelihood` method which can be used
along with functions from `scipy.optimize` to optimize hyperparameters.


Gaussian processes
==================
To understand what a Gaussian process is, let's first consider a random vector
:math:`\mathbf{u}` which has a multivariate normal distribution with mean
:math:`\\bar{\mathbf{u}}` and covariance matrix :math:`\mathbf{C}`. That is to
say, each element :math:`u_i` of :math:`\mathbf{u}` is a normally distributed
random variable with mean :math:`\\bar{u}_i` and covariance :math:`C_{ij}` with
element :math:`u_j`. Each element also has context (e.g., time or position)
denoted as :math:`x_i`. A Gaussian process is the continuous analogue to the
multivariate normal vector, where the context for a Gaussian process is a
continuous variable :math:`x`, rather than the discrete variable :math:`x_i`. A
Gaussian process :math:`u_o` is defined in terms of a mean *function*
:math:`\\bar{u}`, and a covariance *function* :math:`C_u`. We write this
definition of :math:`u_o` more concisely as

.. math::
    u_o \\sim \\mathcal{GP}\\left(\\bar{u},C_u\\right).

Analogous to each element of the random vector :math:`\mathbf{u}`, the Gaussian
process at :math:`x`, denoted as :math:`u_o(x)`, is a normally distributed
random variable with mean :math:`\\bar{u}(x)` and covariance :math:`C_u(x, x')`
with :math:`u_o(x')`.

In this module, we adopt a more general definition of a Gaussian process by
incorporating basis functions. These basis functions are added to Gaussian
processes to account for arbitrary shifts or trends in the data that we are
trying to model. To be more precise, we consider a Gaussian process
:math:`u(x)` to be the combination of :math:`u_o(x)`, a *proper* Gaussian
process, and a set of :math:`m` basis functions,
:math:`\mathbf{p}_u(x) = \{p_i(x)\}_{i=1}^m`, whose coefficients,
:math:`\{c_i\}_{i=1}^m`, are completely unknown (i.e., they have infinite
variance). We then express :math:`u(x)` as

.. math::
    u(x) = u_o(x) + \sum_{i=1}^m c_i p_i(x).

When we include these basis functions, the Gaussian process :math:`u(x)`
becomes improper because it has infinite variance. So when we refer to the
covariance function for a Gaussian process :math:`u(x)`, we are actually
referring to the covariance function for its proper component :math:`u_o(x)`.

Throughout this module we will define a Gaussian process `u(x)` in terms of its
mean function :math:`\\bar{u}(x)`, its covariance function :math:`C_u(x, x')`,
as well as its basis functions :math:`\mathbf{p}_u(x)`.


Operations on Gaussian processes
================================
Gaussian processes can be constructed from other Gaussian processes. There are
four implemented operations on Gaussian processes that result in a new Gaussian
process: addition, scaling, differentiation, and conditioning.


Addition
--------
Two uncorrelated Gaussian processes, :math:`u` and :math:`v`, can be added as

.. math::
    u(x) + v(x) = z(x)

where the mean, covariance, and basis functions for :math:`z` are

.. math::
    \\bar{z}(x) = \\bar{u}(x) + \\bar{v}(x),

.. math::
    C_z(x,x') = C_u(x,x') + C_v(x,x'),

and

.. math::
    \mathbf{p}_z(x) = \mathbf{p}_u(x) \cup \mathbf{p}_v(x).


Scaling
-------
A Gaussian process can be scaled by a constant as

.. math::
    cu(x) = z(x)

where

.. math::
    \\bar{z}(x) = c\\bar{u}(x),

.. math::
    C_z(x,x') = c^2C_u(x,x'),

and

.. math::
    \mathbf{p}_z(x) = \mathbf{p}_u(x).


Differentiation
---------------
A Gaussian process can be differentiated with respect to :math:`x_i` as

.. math::
    \\frac{\partial}{\partial x_i} u(x) = z(x),

where

.. math::
    \\bar{z}(x) = \\frac{\partial}{\partial x_i}\\bar{u}(x),

.. math::
    C_z(x,x') = \\frac{\partial^2}{\partial x_i \partial x_i'} C_u(x,x'),

and

.. math::
    \mathbf{p}_z(x) = \\left\{\\frac{\partial}{\partial x_i} p_k(x)
                      \mid p_k(x) \in \mathbf{p}_u(x)\\right\}


Conditioning
------------
A Gaussian process can be conditioned with :math:`q` noisy observations of
:math:`u(x)`, :math:`\mathbf{d}=\{d_i\}_{i=1}^q`, which have been made at
locations :math:`\mathbf{y}=\{y_i\}_{i=1}^q`. These observations have noise
with zero mean and covariance described by :math:`\mathbf{C_d}`. The
conditioned Gaussian process is

.. math::
    u(x) | \mathbf{d} = z(x)

where

.. math::
    \\bar{z}(x) = \\bar{u}(x) +
                  \mathbf{k}(x,\mathbf{y})
                  \mathbf{K}(\mathbf{y})^{-1}
                  \mathbf{r},

.. math::
    C_{z}(x,x') = C_u(x,x') -
                  \mathbf{k}(x,\mathbf{y})
                  \mathbf{K}(\mathbf{y})^{-1}
                  \mathbf{k}(x',\mathbf{y})^T,

and

.. math::
    \mathbf{p}_z(x) = \emptyset.

In the above equations we use the augmented covariance matrices,
:math:`\mathbf{k}` and :math:`\mathbf{K}`, whose entries are

.. math::
    \mathbf{k}(x,\mathbf{y}) =
    \\left[
    \\begin{array}{cc}
        \\left[C_u(x,y_i)\\right]_{y_i \in \mathbf{y}}
        & \mathbf{p}_u(x) \\\\
    \\end{array}
    \\right]

and

.. math::
    \mathbf{K}(\mathbf{y}) =
    \\left[
    \\begin{array}{cc}
        \mathbf{C_d} + \\left[C_u(y_i,y_j)\\right]_
        {y_i,y_j \in \mathbf{y}\\times\mathbf{y}}
        & [\mathbf{p}_u(y_i)]_{y_i \in \mathbf{y}} \\\\
        [\mathbf{p}_u(y_i)]^T_{y_i \in \mathbf{y}}
        & \mathbf{0}    \\\\
    \\end{array}
    \\right].

We also used the residual vector, :math:`\mathbf{r}`, whose entries are

.. math::
    \mathbf{r} = 
    \\left[
    \\begin{array}{c}
        \\left([d_i - \\bar{u}(y_i)]_{i=1}^q\\right)^T \\\\
        \mathbf{0}
    \\end{array}
    \\right].

Note that there are no basis functions in :math:`z` because it is assumed that
there is enough data in :math:`\mathbf{d}` to constrain the basis functions in
:math:`u`. If :math:`\mathbf{d}` is not sufficiently informative then
:math:`\mathbf{K}(\mathbf{y})` will not be invertible. A necessary but not
sufficient condition for :math:`\mathbf{K}(\mathbf{y})` to be invertible is
that :math:`q \geq m`.


Some commonly used Gaussian processes
=====================================
The `GaussianProcess` class is quite general as it can be instantiated with any
user-specified mean function, covariance function, or set of basis functions.
However, supplying these requisite functions can be laborious. This module
contains several constructors to simplify instantiating some commonly used
types of Gaussian processes. The types of Gaussian processes which have
constructors are listed below.


Isotropic Gaussian Processes
----------------------------
An isotropic Gaussian process has zero mean and a covariance function which can
be written as a function of :math:`r = ||x - x'||_2` and a shape parameters
:math:`\epsilon`,

.. math::
    C_u(x,x') = \sigma^2 \phi(r\ ; \epsilon),

where :math:`\phi(r\ ; \epsilon)` is a positive definite radial basis function.
One common choice for :math:`\phi` is the squared exponential function,

.. math::
    \phi(r\ ;\epsilon) = \exp\\left(\\frac{-r^2}{\epsilon^2}\\right),

which has the useful property of being infinitely differentiable.

An instance of a `GaussianProcess` with an isotropic covariance function can be
created with the function `gpiso`.


Gaussian Process with a Gibbs covariance function
-------------------------------------------------
A Gaussian process with a Gibbs covariance function is useful because, unlike
for isotropic Gaussian processes, it can have a spatially variable lengthscale.
Given some user-specified lengthscale function :math:`\ell_d(x)`, which gives
the lengthscale at :math:`x \in \mathbb{R}^D` along dimension :math:`d`, the
Gibbs covariance function is

.. math::
    C_u(x, x') =
    \sigma^2
    \prod_{d=1}^D \\left(
    \\frac{2 \ell_d(x) \ell_d(x')}{\ell_d(x)^2 + \ell_d(x')^2}
    \\right)^{1/2}
    \exp\\left(-\sum_{d=1}^D
    \\frac{(x_d - x_d')^2}{\ell_d(x)^2 + \ell_d(x')^2}
    \\right).

An instance of a `GaussianProcess` with a Gibbs covariance function can be
created with the function `gpgibbs`.


Gaussian Process with mononomial basis functions
------------------------------------------------
Polynomials are often added to Gaussian processes to improve their ability to
describe offsets and trends in data. The function `gppoly` is used to create a
`GaussianProcess` with zero mean, zero covariance, and a set of monomial basis
function that span the space of all polynomials with some degree, :math:`d`.
For example, if :math:`x \in \mathbb{R}^2` and :math:`d=1`, then the monomial
basis functions would be

.. math::
    \mathbf{p}_u(x) = \{1,x_1,x_2\}.


Examples
========
Here we provide a basic example that demonstrates creating a `GaussianProcess`
and performing GPR. Suppose we have 5 scalar valued observations `d` that were
made at locations `x`, and we want interpolate these observations with GPR

>>> x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
>>> d = [2.3, 2.2, 1.7, 1.8, 2.4]

First we define our prior for the underlying function that we want to
interpolate. We assume an isotropic `GaussianProcess` with a squared
exponential covariance function and the parameter :math:`\mu=0.0`,
:math:`\sigma^2=1.0` and :math:`\epsilon=0.5`.

>>> from rbf.gproc import gpiso
>>> gp_prior = gpiso('se', eps=0.5, var=1.0)

We also want to include an unknown constant offset to our prior model, which is
done with the command

>>> from rbf.gproc import gppoly
>>> gp_prior += gppoly(0)

Now we condition the prior with the observations to form the posterior

>>> gp_post = gp_prior.condition(x, d)

We can now evaluate the mean and covariance of the posterior anywhere using the
`mean` or `covariance` method. We can also evaluate just the mean and standard
deviation by calling the instance

>>> m, s = gp_post([[0.5], [1.5], [2.5], [3.5]])


References
==========
[1] Rasmussen, C., and Williams, C., Gaussian Processes for Machine Learning.
The MIT Press, 2006.

'''
import logging
import warnings
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


def zero_mean(x, diff):
    '''Mean function that returns zeros.'''
    return np.zeros((x.shape[0],), dtype=float)


def zero_variance(x, diff):
    '''Variance function that returns zeros.'''
    return np.zeros((x.shape[0],), dtype=float)


def zero_covariance(x1, x2, diff1, diff2):
    '''Covariance function that returns zeros.'''
    return np.zeros((x1.shape[0], x2.shape[0]), dtype=float)


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
        Mean vector.

    cov : (N, N) array or sparse matrix
        Covariance matrix.

    use_cholesky : bool, optional
        Whether to use the Cholesky decomposition or eigenvalue decomposition.
        The former is faster but fails when `cov` is not numerically positive
        definite.

    count : int, optional
        Number of samples to draw.

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


def log_likelihood(d, mu, cov, vecs=None):
    '''
    Returns the log likelihood of observing `d` from a multivariate normal
    distribution with mean `mu` and covariance `cov`.

    When `vecs` is specified, the restricted log likelihood is returned. The
    restricted log likelihood is the probability of observing `R.dot(d)` from a
    normally distributed random vector with mean `R.dot(mu)` and covariance
    `R.dot(sigma).dot(R.T)`, where `R` is a matrix with rows that are
    orthogonal to the columns of `vecs`. See [1] or [2] for more information.

    Parameters
    ----------
    d : (N,) array
        Observation vector.

    mu : (N,) array
        Mean vector.

    cov : (N, N) array or sparse matrix
        Covariance matrix.

    vecs : (N, M) array, optional
        Unconstrained basis vectors.

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

    if vecs is None:
        vecs = np.zeros((n, 0), dtype=float)
    else:
        vecs = np.asarray(vecs, dtype=float)
        assert_shape(vecs, (n, None), 'vecs')

    m = vecs.shape[1]

    A = PosDefSolver(cov)
    B = A.solve_L(vecs)
    C = PosDefSolver(B.T.dot(B))
    D = PosDefSolver(vecs.T.dot(vecs))

    a = A.solve_L(d - mu)
    b = C.solve_L(B.T.dot(a))

    out = 0.5*(D.log_det() -
               A.log_det() -
               C.log_det() -
               a.T.dot(a) +
               b.T.dot(b) -
               (n-m)*np.log(2*np.pi))
    return out


def outliers(d, dsigma, pcov, pmu=None, pvecs=None, tol=4.0, maxitr=50):
    '''
    Uses a data editing algorithm to identify outliers in `d`. Outliers are
    considered to be the data that are abnormally inconsistent with a
    multivariate normal distribution with mean `pmu`, covariance `pcov`, and
    basis vectors `pvecs`.

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

    pcov : (N, N) array or sparse matrix
        Covariance of the prior at the observation points.

    pmu : (N,) float array, optional
        Mean of the prior at the observation points. Defaults to zeros.

    pvecs : (N, P) float array, optional
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
    (N,) bool array
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

    if pvecs is None:
        pvecs = np.zeros((n, 0), dtype=float)
    else:
        pvecs = np.asarray(pvecs, dtype=float)
        assert_shape(pvecs, (n, None), 'pvecs')

    # Total number of outlier detection iterations completed thus far
    itr = 0
    inliers = np.ones(n, dtype=bool)
    while True:
        LOGGER.debug(
            'Starting iteration %d of outlier detection.' % (itr+1)
            )
        # Remove rows and cols corresponding to the outliers
        pcov_i = pcov[:, inliers][inliers, :]
        pmu_i = pmu[inliers]
        pvecs_i = pvecs[inliers]
        d_i = d[inliers]
        dsigma_i = dsigma[inliers]
        if sp.issparse(pcov):
            pcov_i = (pcov_i + sp.diags(dsigma_i**2)).tocsc()
        else:
            pcov_i = pcov_i + np.diag(dsigma_i**2)

        # Find the mean of the posterior
        solver = PartitionedPosDefSolver(pcov_i, pvecs_i)
        v1, v2 = solver.solve(d_i - pmu_i)
        fit = pmu + pcov[:, inliers].dot(v1) + pvecs.dot(v2)

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


def _condition(gp, y, d, dcov, dvecs, ddiff, build_inverse):
    '''
    Returns a conditioned `GaussianProcess`.
    '''
    if gp._mean is None:
        prior_mean = zero_mean
    else:
        prior_mean = gp._mean

    if gp._covariance is None:
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
    vecs = prior_basis(y, ddiff)
    if dvecs.shape[1] != 0:
        vecs = np.hstack((vecs, dvecs))

    solver = PartitionedPosDefSolver(
        cov, vecs, build_inverse=build_inverse
        )

    # precompute these vectors which are used for `posterior_mean`
    v1, v2 = solver.solve(res)

    del res, cov, vecs

    def posterior_mean(x, diff):
        mu_x = prior_mean(x, diff)
        cov_xy = prior_covariance(x, y, diff, ddiff)
        vecs_x = prior_basis(x, diff)
        if dvecs.shape[1] != 0:
            pad = np.zeros((x.shape[0], dvecs.shape[1]), dtype=float)
            vecs_x = np.hstack((vecs_x, pad))

        out = mu_x + cov_xy.dot(v1) + vecs_x.dot(v2)
        return out

    def posterior_covariance(x1, x2, diff1, diff2):
        cov_x1x2 = prior_covariance(x1, x2, diff1, diff2)
        cov_x1y = prior_covariance(x1, y, diff1, ddiff)
        cov_x2y = prior_covariance(x2, y, diff2, ddiff)
        vecs_x1 = prior_basis(x1, diff1)
        vecs_x2 = prior_basis(x2, diff2)
        if dvecs.shape[1] != 0:
            pad = np.zeros((x1.shape[0], dvecs.shape[1]), dtype=float)
            vecs_x1 = np.hstack((vecs_x1, pad))

            pad = np.zeros((x2.shape[0], dvecs.shape[1]), dtype=float)
            vecs_x2 = np.hstack((vecs_x2, pad))

        m1, m2 = solver.solve(cov_x2y.T, vecs_x2.T)
        out = cov_x1x2 - cov_x1y.dot(m1) - vecs_x1.dot(m2)
        # `out` may either be a matrix or array depending on whether cov_x1x2
        # is sparse or dense. Make the output consistent by converting to array
        out = np.asarray(out)
        return out

    def posterior_variance(x, diff):
        var_x = prior_variance(x, diff)
        cov_xy = prior_covariance(x, y, diff, ddiff)
        vecs_x = prior_basis(x, diff)
        if dvecs.shape[1] != 0:
            pad = np.zeros((x.shape[0], dvecs.shape[1]), dtype=float)
            vecs_x = np.hstack((vecs_x, pad))

        m1, m2 = solver.solve(cov_xy.T, vecs_x.T)
        # Efficiently get the diagonals of C_xy.dot(mat1) and p_x.dot(mat2)
        if sp.issparse(cov_xy):
            diag1 = cov_xy.multiply(m1.T).sum(axis=1).A[:, 0]
        else:
            diag1 = np.einsum('ij, ji -> i', cov_xy, m1)

        diag2 = np.einsum('ij, ji -> i', vecs_x, m2)
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
    Class for performing basic operations with Gaussian processes. A Gaussian
    process is stochastic process defined by a mean function, a covariance
    function, and a set of basis functions.

    Parameters
    ----------
    mean : callable, optional
        Function which returns either the mean of the Gaussian process at `x`
        or a specified derivative of the mean at `x`. This has the call
        signature

        `out = mean(x)`

        or

        `out = mean(x, diff)`,

        where `x` is an (N, D) float array, `diff` is a (D,) int array, and
        `out` is an (N,) float array.

    covariance : callable, optional
        Function which returns either the covariance of the Gaussian process
        between points `x1` and `x2` or the covariance of the specified
        derivatives of the Gaussian process between points `x1` and `x2`. This
        has the call signature

        `out = covariance(x1, x2)`

        or

        `out = covariance(x1, x2, diff1, diff2)`,

        where `x1` is an (N, D) float array, `x2` is an (M, D) float array,
        `diff1` and `diff2` are (D,) int arrays, and `out` is an (N, M) float
        array or sparse matrix.

    basis : callable, optional
        Function which returns either the basis functions evaluated at `x` or
        the specified derivative of the basis functions evaluated at `x`. This
        has the call signature

        `out = basis(x)`

        or

        `out = basis(x, diff)`,

        where `x` is an (N, D) float array, `diff` is a (D,) int array, and
        `out` is an (N, P) float array.

    variance : callable, optional
        A function that returns the variance of the Gaussian process or its
        derivative at `x`. The has the call signature

        `out = variance(x)`

        or

        `out = variance(x, diff)`,

        where `x` is an (N, D) float array, `diff` is a (D,) int array, and
        `out` is an (N,) float array. If given, this should be more efficient
        than evaluating `covariance` and taking its diagonals.

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
        '''Returns a scaled `GaussianProcess`.'''
        c = float(c)
        out = _scale(self, c)
        return out

    def differentiate(self, d):
        '''Returns a differentiated `GaussianProcess`.'''
        d = np.asarray(d, dtype=int)
        assert_shape(d, (self.dim,), 'd')
        out = _differentiate(self, d)
        return out

    def condition(self, y, d,
                  dcov=None,
                  dvecs=None,
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

        dcov : (N, N) array or sparse matrix, optional
            Covariance of the data noise. Defaults to a dense array of zeros.

        dvecs : (N, P) array, optional
            Data noise basis vectors. The data noise is assumed to contain some
            unknown linear combination of the columns of `dvecs`.

        ddiff : (D,) int array, optional
            Derivative of the observations. For example, use (1,) if the
            observations are 1-D and should constrain the slope.

        build_inverse : bool, optional
            Whether to construct the inverse matrices rather than just the
            factors.

        Returns
        -------
        GaussianProcess

        '''
        y = np.asarray(y, dtype=float)
        assert_shape(y, (None, self.dim), 'y')
        n, dim = y.shape

        d = np.asarray(d, dtype=float)
        assert_shape(d, (n,), 'd')

        if dcov is None:
            dcov = np.zeros((n, n), dtype=float)
        else:
            dcov = as_sparse_or_array(dcov)
            assert_shape(dcov, (n, n), 'dcov')

        if dvecs is None:
            dvecs = np.zeros((n, 0), dtype=float)
        else:
            dvecs = np.asarray(dvecs, dtype=float)
            assert_shape(dvecs, (n, None), 'dvecs')

        if ddiff is None:
            ddiff = np.zeros(dim, dtype=int)
        else:
            ddiff = np.asarray(ddiff, dtype=int)
            assert_shape(ddiff, (dim,), 'ddiff')

        out = _condition(
            self, y, d, dcov, dvecs, ddiff,
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
        Returns the covariance matrix of the Gaussian process.

        Parameters
        ----------
        x1, x2 : (N, D) and (M, D) array
            Evaluation points.

        diff1, diff2 : (D,) int array
            Derivative specifications.

        Returns
        -------
        (N, M) array or sparse matrix

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

    def log_likelihood(self, y, d, dcov=None, dvecs=None):
        '''
        Returns the log likelihood of drawing the observations `d` from the
        Gaussian process. The observations could potentially have noise which
        is described by `dcov` and `dvecs`. If the Gaussian process contains
        any basis functions or if `dvecs` is specified, then the restricted
        log likelihood is returned.

        Parameters
        ----------
        y : (N, D) array
            Observation points.

        d : (N,) array
            Observed values at `y`.

        dcov : (N, N) array or sparse matrix, optional
            Data covariance. If not given, this will be a dense matrix of
            zeros.

        dvecs : (N, P) float array, optional
            Basis vectors for the noise. The data noise is assumed to contain
            some unknown linear combination of the columns of `dvecs`.

        Returns
        -------
        float

        '''
        y = np.asarray(y, dtype=float)
        assert_shape(y, (None, self.dim), 'y')
        n, dim = y.shape

        d = np.asarray(d, dtype=float)
        assert_shape(d, (n,), 'd')

        if dcov is None:
            dcov = np.zeros((n, n), dtype=float)
        else:
            dcov = as_sparse_or_array(dcov)
            assert_shape(dcov, (n, n), 'dcov')

        if dvecs is None:
            dvecs = np.zeros((n, 0), dtype=float)
        else:
            dvecs = np.asarray(dvecs, dtype=float)
            assert_shape(dvecs, (n, None), 'dvecs')

        mu = self.mean(y)
        cov = as_sparse_or_array(dcov + self.covariance(y, y))
        vecs = np.hstack((self.basis(y), dvecs))

        out = log_likelihood(d, mu, cov, vecs=vecs)
        return out

    def sample(self, x, use_cholesky=False, count=None):
        '''
        Draws a random sample from the Gaussian process.
        '''
        mu = self.mean(x)
        cov = self.covariance(x, x)
        out = sample(mu, cov, use_cholesky=use_cholesky, count=count)
        return out


    def outliers(self, x, d, dsigma, tol=4.0, maxitr=50):
        '''
        Identifies values in `d` that are abnormally inconsistent with the the
        Gaussian process

        Parameters
        ----------
        x : (N, D) float array
            Observations locations.

        d : (N,) float array
            Observations.

        dsigma : (N,) float array
            One standard deviation uncertainty on the observations.

        tol : float, optional
            Outlier tolerance. Smaller values make the algorithm more likely to
            identify outliers. A good value is 4.0 and this should not be set
            any lower than 2.0.

        maxitr : int, optional
            Maximum number of iterations.

        Returns
        -------
        (N,) bool array
            Array indicating which data are outliers

        '''
        x = np.asarray(x, dtype=float)
        assert_shape(x, (None, self.dim), 'x')
        n, dim = x.shape

        d = np.asarray(d, dtype=float)
        assert_shape(d, (n,), 'd')

        dsigma = np.asarray(dsigma, dtype=float)
        assert_shape(dsigma, (n,), 'dsigma')

        pcov = self.covariance(x, x)
        pmu = self.mean(x)
        pvecs = self.basis(x)
        out = outliers(
            d, dsigma, pcov, pmu=pmu, pvecs=pvecs, tol=tol, maxitr=maxitr
            )
        return out


    def is_positive_definite(self, x):
        '''
        Tests if the covariance function evaluated at `x` is positive definite.
        '''
        cov = self.covariance(x, x)
        out = is_positive_definite(cov)
        return out


def gpiso(phi, eps, var, dim=None):
    '''
    Creates an isotropic Gaussian process which has a covariance function that
    is described by a radial basis function.

    Parameters
    ----------
    phi : str or RBF instance
        Radial basis function describing the covariance function.

    eps : float
        Shape parameter.

    var : float
        Variance.

    dim : int, optional
        Fixes the spatial dimensions of the domain.

    Returns
    -------
    GaussianProcess

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
        coeff = var*(-1)**sum(diff2)
        out = coeff*phi(x1, x2, eps=eps, diff=diff)
        return out

    def isotropic_variance(x, diff):
        coeff = var*(-1)**sum(diff)
        value = coeff*phi.center_value(eps=eps, diff=2*diff)
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
    Returns a Gaussian process consisting of monomial basis functions.

    Parameters
    ----------
    order : int
        Order of the polynomial spanned by the basis functions.

    dim : int, optional
        Fixes the spatial dimensions of the domain.

    Returns
    -------
    GaussianProcess

    '''
    def polynomial_basis(x, diff):
        powers = rbf.poly.monomial_powers(order, x.shape[1])
        out = rbf.poly.mvmonos(x, powers, diff)
        return out

    out = GaussianProcess(basis=polynomial_basis, dim=dim, wrap=False)
    return out


def gpgibbs(lengthscale, sigma, delta=1e-4):
    '''
    Returns a Gaussian process with a Gibbs covariance function. The Gibbs
    kernel has a spatially varying lengthscale.

    Parameters
    ----------
    lengthscale: function
        Function that takes an (N, D) array of positions and returns an (N, D)
        array indicating the lengthscale along each dimension at those
        positions.

    sigma: float
        Standard deviation of the Gaussian process.

    delta: float, optional
        Finite difference spacing to use when calculating the derivative of the
        `GaussianProcess`. An analytical solution for the derivative is not
        available because the derivative of the `lengthscale` function is
        unknown.

    Returns
    -------
    GaussianProcess

    '''
    @covariance_differentiator(delta)
    def gibbs_covariance(x1, x2):
        '''
        covariance function for the Gibbs Gaussian process.
        '''
        dim = x1.shape[1]
        lsx1 = lengthscale(x1)
        lsx2 = lengthscale(x2)

        # sanitize the output for `lengthscale`
        lsx1 = np.asarray(lsx1, dtype=float)
        lsx2 = np.asarray(lsx2, dtype=float)
        assert_shape(lsx1, x1.shape, 'lengthscale(x1)')
        assert_shape(lsx2, x2.shape, 'lengthscale(x2)')

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

    out = GaussianProcess(covariance=gibbs_covariance, wrap=False)
    return out
