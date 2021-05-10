'''
THIS MODULE HAS BEEN REPLACED BY `gproc.py` AND IT REMAINS HERE FOR LEGACY
PURPOSES

This module defines a class, `GaussianProcess`, which is an
abstraction that allows one to easily work with Gaussian processes.
One main use for the `GaussianProcess` class is Gaussian process
regression (GPR). GPR is also known as Kriging or Least Squares
Collocation.  It is a technique for constructing a continuous function
from discrete observations by incorporating a stochastic prior model
for the underlying function.  GPR is performed with the `condition`
method of a `GaussianProcess` instance. In addition to GPR, the
`GaussianProcess` class can be used for basic arithmetic with Gaussian
processes and for generating random samples of a Gaussian process.

There are several existing python packages for Gaussian processes (See
www.gaussianprocess.org for an updated list of packages). This module
was written because existing software lacked support for 1) Gaussian
processes with added basis functions 2) analytical differentiation of
Gaussian processes and 3) conditioning a Gaussian process with
derivative constraints. Other software packages have a strong focus on
optimizing hyperparameters based on data likelihood. This module does
not include any optimization routines and hyperparameters are always
explicitly specified by the user. However, the `GaussianProcess` class
contains the `likelihood` method which can be used with functions from
`scipy.optimize` to construct a hyperparameter optimization routine.


Gaussian processes
==================
To understand what a Gaussian process is, let's first consider a
random vector :math:`\mathbf{u}` which has a multivariate normal
distribution with mean :math:`\\bar{\mathbf{u}}` and covariance matrix
:math:`\mathbf{C}`. That is to say, each element :math:`u_i` of
:math:`\mathbf{u}` is a normally distributed random variable
with mean :math:`\\bar{u}_i` and covariance :math:`C_{ij}` with
element :math:`u_j`. Each element also has context (e.g., time or
position) denoted as :math:`x_i`. A Gaussian process is the continuous
analogue to the multivariate normal vector, where the context for a
Gaussian process is a continuous variable :math:`x`, rather than the
discrete variable :math:`x_i`. A Gaussian process :math:`u_o` is
defined in terms of a mean *function* :math:`\\bar{u}`, and a
covariance *function* :math:`C_u`. We write this definition of
:math:`u_o` more concisely as

.. math::
  u_o \\sim \\mathcal{GP}\\left(\\bar{u},C_u\\right).

Analogous to each element of the random vector :math:`\mathbf{u}`, the
Gaussian process at :math:`x`, denoted as :math:`u_o(x)`, is a
normally distributed random variable with mean :math:`\\bar{u}(x)` and
covariance :math:`C_u(x, x')` with :math:`u_o(x')`.

In this module, we adopt a more general definition of a Gaussian
process by incorporating basis functions. These basis functions are
added to Gaussian processes to account for arbitrary shifts or trends
in the data that we are trying to model. To be more precise, we
consider a Gaussian process :math:`u(x)` to be the combination of
:math:`u_o(x)`, a *proper* Gaussian process, and a set of :math:`m`
basis functions, :math:`\mathbf{p}_u(x) = \{p_i(x)\}_{i=1}^m`, whose
coefficients, :math:`\{c_i\}_{i=1}^m`, have infinite variance. We then
express :math:`u(x)` as

.. math::
  u(x) = u_o(x) + \sum_{i=1}^m c_i p_i(x).

When we include these basis functions, the Gaussian process
:math:`u(x)` becomes improper because it has infinite variance. So
when we refer to the covariance function for a Gaussian process
:math:`u(x)`, we are actually referring to the covariance function for
its proper component :math:`u_o(x)`.

Throughout this module we will define a Gaussian process `u(x)` in
terms of its mean function :math:`\\bar{u}(x)`, its covariance
function :math:`C_u(x, x')`, as well as its basis functions
:math:`\mathbf{p}_u(x)`.

We consider five operations on Gaussian processes: addition,
subtraction, scaling, differentiation, and conditioning. Each
operation produces another Gaussian process which possesses the same
five operations. These operations are described below.


Operations on Gaussian processes
================================

Addition
--------
Two uncorrelated Gaussian processes, :math:`u` and :math:`v`, can be
added as

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

Two `GaussianProcess` instances can be added with the `add` method or
the `+` operator.

Subtraction
-----------
A Gaussian process can be subtracted from another Gaussian processes
as

.. math::
  u(x) - v(x) = z(x)

where

.. math::
  \\bar{z}(x) = \\bar{u}(x) - \\bar{v}(x),

.. math::
  C_z(x,x') = C_u(x,x') + C_v(x,x'),

and

.. math::
  \mathbf{p}_z(x) = \mathbf{p}_u(x) \cup \mathbf{p}_v(x).

Two `GaussianProcess` instances can be subtracted with the `subtract`
method or the `-` operator.

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

A `GaussianProcess` instance can be scaled with the `scale` method or
the `*` operator.

Differentiation
---------------
A Gaussian process can be differentiated with respect to :math:`x_i`
as

.. math::
  \\frac{\partial}{\partial x_i} u(x) = z(x),

where

.. math::
  \\bar{z}(x) = \\frac{\partial}{\partial x_i}\\bar{u}(x),

.. math::
  C_z(x,x') = \\frac{\partial^2}{\partial x_i \partial x_i'}
              C_u(x,x'),

and

.. math::
  \mathbf{p}_z(x) = \\left\{\\frac{\partial}{\partial x_i} p_k(x)
                    \mid p_k(x) \in \mathbf{p}_u(x)\\right\}

A `GaussianProcess` instance can be differentiated with the
`differentiate` method.

Conditioning
------------
A Gaussian process can be conditioned with :math:`q` noisy
observations of :math:`u(x)`, :math:`\mathbf{d}=\{d_i\}_{i=1}^q`,
which have been made at locations :math:`\mathbf{y}=\{y_i\}_{i=1}^q`.
These observations have noise with zero mean and covariance described
by :math:`\mathbf{C_d}`. The conditioned Gaussian process is

.. math::
  u(x) | \mathbf{d} = z(x)

where

.. math::
  \\bar{z}(x) = \\bar{u}(x) +
                \mathbf{k}(x,\mathbf{y})
                \mathbf{K}(\mathbf{y})^{-1}
                \mathbf{r}^*,

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

We define the residual vector as

.. math::
  \mathbf{r} = \\left([d_i - \\bar{u}(y_i)]_{i=1}^q\\right)^T

and :math:`\mathbf{r}^*` is the residual vector which has been
suitably padded with zeros. Note that there are no basis functions in
:math:`z` because it is assumed that there is enough data in
:math:`\mathbf{d}` to constrain the basis functions in :math:`u`. If
:math:`\mathbf{d}` is not sufficiently informative then
:math:`\mathbf{K}(\mathbf{y})` will not be invertible. A necessary but
not sufficient condition for :math:`\mathbf{K}(\mathbf{y})` to be
invertible is that :math:`q \geq m`.

A `GaussianProcess` instance can be conditioned with the `condition`
method.

Some commonly used Gaussian processes
=====================================
The `GaussianProcess` class is quite general as it can be instantiated
with any user-specified mean function, covariance function, or set of
basis functions. However, supplying these requisite functions can be
laborious. This module contains several constructors to simplify
instantiating some commonly used types of Gaussian processes. The
types of Gaussian processes which have constructors are listed below.


Isotropic Gaussian Processes
----------------------------
An isotropic Gaussian process has a constant mean and a covariance
function which can be written as a function of
:math:`r = ||x - x'||_2`. To put more explicitly, an isotropic
Gaussian processes has the mean function

.. math::
  \\bar{u}(x) = \mu,

and the covariance function

.. math::
  C_u(x,x') = \sigma^2 \phi(r\ ; \epsilon),

Where :math:`\phi(r\ ; \epsilon)` is a positive definite radial basis
function with shape parameter :math:`\epsilon`. One common choice for
:math:`\phi` is the squared exponential function,

.. math::
  \phi(r\ ;\epsilon) = \exp\\left(\\frac{-r^2}{\epsilon^2}\\right),

which has the useful property of being infinitely differentiable. An
instance of an isotropic `GaussianProcess` can be created with the
function `gpiso`. A `GaussianProcess` with a squared exponential
covariance function can be created with the function `gpse`.


Gaussian Process with a Gibbs covariance function
-------------------------------------------------
A Gaussian process with a Gibbs covariance function is useful because,
unlike for isotropic Gaussian processes, it can have a spatially
variable lengthscale. Given some user-specified lengthscale function
:math:`\ell_d(x)`, which gives the lengthscale at :math:`x \in
\mathbb{R}^D` along dimension :math:`d`, the Gibbs covariance function
is

.. math::
    C_u(x, x') =
    \sigma^2
    \prod_{d=1}^D \\left(
    \\frac{2 \ell_d(x) \ell_d(x')}{\ell_d(x)^2 + \ell_d(x')^2}
    \\right)^{1/2}
    \exp\\left(-\sum_{d=1}^D
    \\frac{(x_d - x_d')^2}{\ell_d(x)^2 + \ell_d(x')^2}
    \\right).

An instance of a `GaussianProcess` with a Gibbs covariance function
can be created with the function `gpgibbs`.


Gaussian Process with mononomial basis functions
------------------------------------------------
Polynomials are often added to Gaussian processes to improve their
ability to describe offsets and trends in data. The function `gppoly`
is used to create a `GaussianProcess` with zero mean, zero covariance,
and a set of monomial basis function that span the space of all
polynomials with some degree, :math:`d`. For example, if :math:`x \in
\mathbb{R}^2` and :math:`d=1`, then the monomial basis functions would
be

.. math::
  \mathbf{p}_u(x) = \{1,x_1,x_2\}.

The function `gpbasis` can be used to create a `GaussianProcess` with
any other type of basis functions.


Examples
========
Here we provide a basic example that demonstrates creating a
`GaussianProcess` and performing GPR. Suppose we have 5 scalar valued
observations `d` that were made at locations `x`, and we want
interpolate these observations with GPR

>>> x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
>>> d = [2.3, 2.2, 1.7, 1.8, 2.4]

First we define our prior for the underlying function that we want to
interpolate. We assume an isotropic `GaussianProcess` with a squared
exponential covariance function and the parameter :math:`\mu=0.0`,
:math:`\sigma^2=1.0` and :math:`\epsilon=0.5`.

>>> from rbf.basis import se
>>> from rbf.gauss import gpiso
>>> gp_prior = gpiso(se, (0.0, 1.0, 0.5))

We also want to include an unknown constant offset to our prior model,
which is done with the command

>>> from rbf.gauss import gppoly
>>> gp_prior += gppoly(0)

Now we condition the prior with the observations to form the posterior

>>> gp_post = gp_prior.condition(x, d)

We can now evaluate the mean and covariance of the posterior anywhere
using the `mean` or `covariance` method. We can also evaluate just the
mean and standard deviation with the `meansd` method.

>>> m, s = gp_post.meansd([[0.5], [1.5], [2.5], [3.5]])


References
==========
[1] Rasmussen, C., and Williams, C., Gaussian Processes for Machine
Learning. The MIT Press, 2006.

'''
import logging
import warnings

import numpy as np
import scipy.sparse as sp

import rbf.poly
import rbf.basis
import rbf.linalg
from rbf.utils import assert_shape, get_arg_count, MemoizeArrayInput
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
    '''The actual decorator'''
    def fout(x, diff):
      '''The returned differentiable mean function'''
      if not any(diff):
        # If no derivatives are specified then return the undifferentiated
        # mean. Make sure the output is a numpy array.
        out = as_array(fin(x))
        return out

      else:
        # get the axis we are differentiating with respect to
        diff_axis = np.argmax(diff)
        # make the perturbations
        x_plus_dx = np.copy(x)
        x_plus_dx[:, diff_axis] += delta
        # make a copy of `diff` and lower the derivative along `diff_axis` by
        # one.
        diff_minus_one = np.copy(diff)
        diff_minus_one[diff_axis] -= 1
        # compute a first order forward finite difference
        out = ( fout(x_plus_dx, diff_minus_one) -
                fout(x,         diff_minus_one) ) / delta
        return out

    return fout

  return _differentiator


def covariance_differentiator(delta):
  '''
  Decorator that makes a covariance function differentiable. The derivatives of
  the covariance function are approximated by finite differences. The
  covariance function must take an (N, D) array and an (M, D) array of
  positions as input. The returned function takes an (N, D) array and an (M, D)
  array of positions and two (D,) array derivative specifications.

  Parameters
  ----------
  delta : float
    step size to use for finite differences

  '''
  def _covariance_differentiator(fin):
    '''The actual decorator'''
    def fout(x1, x2, diff1, diff2):
      '''The returned differentiable mean function'''
      if (not any(diff1)) & (not any(diff2)):
        # If no derivatives are specified then return the undifferentiated
        # covariance.
        return as_sparse_or_array(fin(x1, x2))

      elif any(diff1):
        # get the axis we are differentiating with respect to
        diff1_axis = np.argmax(diff1)
        # make the perturbations
        x1_plus_dx = np.copy(x1)
        x1_plus_dx[:, diff1_axis] += delta
        # make a copy of `diff1` and lower the derivative along `diff1_axis` by
        # one.
        diff1_minus_one = np.copy(diff1)
        diff1_minus_one[diff1_axis] -= 1
        # compute a first order forward finite difference
        out = ( fout(x1_plus_dx, x2, diff1_minus_one, diff2) -
                fout(x1,         x2, diff1_minus_one, diff2) ) / delta
        return out

      else:
        # any(diff2) == True
        # get the axis we are differentiating with respect to
        diff2_axis = np.argmax(diff2)
        # make the perturbations
        x2_plus_dx = np.copy(x2)
        x2_plus_dx[:, diff2_axis] += delta
        # make a copy of `diff2` and lower the derivative along `diff2_axis` by
        # one.
        diff2_minus_one = np.copy(diff2)
        diff2_minus_one[diff2_axis] -= 1
        # compute a first order forward finite difference
        out = ( fout(x1, x2_plus_dx, diff1, diff2_minus_one) -
                fout(x1,         x2, diff1, diff2_minus_one) ) / delta
        return out

    return fout

  return _covariance_differentiator


def _combined_dim(dim1, dim2):
  '''
  Returns the dimensionality of a Gaussian process formed by combining two
  Gaussian processes with dimensions `dim1` and `dim2`. The dimensionality can
  be an `int` or `None` indicating that it is unspecified
  '''
  # If both dimensions are unspecified, return None
  if (dim1 is None) & (dim2 is None):
    return None

  # At least one dimension is specified. If only one dimension is specified
  # return that
  elif dim1 is None:
    return dim2

  elif dim2 is None:
    return dim1

  # both dim1 and dim2 are specified. If they are not equal raise an error
  elif dim1 == dim2:
    return dim1

  else:
    raise ValueError(
        'The `GaussianProcess` instances have inconsistent spatial dimensions')


def _as_covariance(sigma):
  '''
  Return `sigma` as a covariance matrix. If `sigma` is a 1-D array then square
  it and make it a scipy sparse diagonal matrix. Otherwise run `sigma` through
  `as_sparse_or_array`
  '''
  if np.ndim(sigma) == 1:
    sigma = np.array(sigma, dtype=float, copy=False)
    sigma = sp.diags(sigma**2).tocsc()

  sigma = as_sparse_or_array(sigma, dtype=float)
  return sigma


def _all_is_finite(A):
  '''
  returns True if all values in `A` are finite. `A` can be a numpy array or a
  scipy sparse matrix.
  '''
  if sp.issparse(A):
    # get all the nonzero entries
    return np.all(np.isfinite(A.data))

  else:
    return np.all(np.isfinite(A))


def _sample(mean, cov, use_cholesky=False, count=None):
  '''
  Draws a random sample from the Gaussian process with the specified mean and
  covariance.
  '''
  if use_cholesky:
    # draw a sample using a cholesky decomposition. This assumes that `cov` is
    # numerically positive definite (i.e. no small negative eigenvalues from
    # rounding error).
    L = PosDefSolver(cov).L()
    if count is None:
      w = np.random.normal(0.0, 1.0, mean.shape[0])
      u = mean + L.dot(w)

    else:
      w = np.random.normal(0.0, 1.0, (mean.shape[0], count))
      u = (mean[:, None] + L.dot(w)).T

  else:
    # otherwise use an eigenvalue decomposition, ignoring negative eigenvalues.
    # If `cov` is sparse then begrudgingly make it dense.
    cov = as_array(cov)
    vals, vecs = np.linalg.eigh(cov)
    keep = (vals > 0.0)
    vals = vals[keep]
    vecs = vecs[:, keep]
    if count is None:
      w = np.random.normal(0.0, np.sqrt(vals))
      u = mean + vecs.dot(w)

    else:
      w = np.random.normal(0.0, np.sqrt(vals[:, None].repeat(count, axis=1)))
      u = (mean[:, None] + vecs.dot(w)).T

  return u


def likelihood(d, mu, sigma, p=None):
  '''
  Returns the log likelihood. If `p` is not specified, then the likelihood is
  the probability of observing `d` from a normally distributed random vector
  with mean `mu` and covariance `sigma`. If `d` is expected to contain some
  unknown linear combination of basis vectors (e.g. a constant offset or linear
  trend), then `p` should be specified with those basis vectors as its columns.
  When `p` is specified, the restricted likelihood is returned. The restricted
  likelihood is the probability of observing `R.dot(d)` from a normally
  distributed random vector with mean `R.dot(mu)` and covariance
  `R.dot(sigma).dot(R.T)`, where `R` is a matrix with rows that are orthogonal
  to the columns of `p`. In other words, if `p` is specified then the component
  of `d` which lies along the columns of `p` will be ignored.

  The restricted likelihood was first described by [1] and it is covered in
  more general reference books such as [2]. Both [1] and [2] are good sources
  for additional information.

  Parameters
  ----------
  d : (N,) array
    observations

  mu : (N,) array
    mean of the random vector

  sigma : (N,) array, (N, N) array, or (N, N) scipy sparse matrix
    If this is an (N,) array then it describes one standard deviation of the
    random vector. If this is an (N, N) array then it describes the
    covariances.

  p : (N, P) array, optional
    Basis vectors. If specified, then `d` is assumed to contain some unknown
    linear combination of the columns of `p`.

  Notes
  -----
  Unlike other functions in this module, if the covariance matrix is not
  numerically positive definite then this function will fail with an error
  rather than trying to coerce it into a positive definite matrix.

  References
  ----------
  [1] Harville D. (1974). Bayesian Inference of Variance Components Using Only
  Error Contrasts. Biometrica.

  [2] Cressie N. (1993). Statistics for Spatial Data. John Wiley & Sons.

  '''
  d = as_array(d, dtype=float)
  assert_shape(d, (None,), 'd')
  # number of observations
  n = d.shape[0]

  mu = as_array(mu, dtype=float)
  assert_shape(mu, (n,), 'mu')

  sigma = _as_covariance(sigma)
  assert_shape(sigma, (n, n), 'sigma')

  if p is None:
    p = np.zeros((n, 0), dtype=float)

  else:
    p = as_array(p, dtype=float)

  assert_shape(p, (n, None), 'p')
  # number of basis vectors
  m = p.shape[1]

  A = PosDefSolver(sigma)
  B = A.solve_L(p)
  C = PosDefSolver(B.T.dot(B))
  D = PosDefSolver(p.T.dot(p))

  a = A.solve_L(d - mu)
  b = C.solve_L(B.T.dot(a))

  out = 0.5*(D.log_det() -
             A.log_det() -
             C.log_det() -
             a.T.dot(a) +
             b.T.dot(b) -
             (n-m)*np.log(2*np.pi))
  return out


def outliers(d, s, mu=None, sigma=None, p=None, tol=4.0, maxitr=50):
  '''
  Uses a data editing algorithm to identify outliers in `d`. Outliers are
  considered to be the data that are abnormally inconsistent with the Gaussian
  process described by `mu` (mean), `sigma` (covariance), and `p` (basis
  vectors). This function can only be used for data with nonzero, uncorrelated
  noise.

  The data editing algorithm first conditions the Gaussian process with the
  observations, then it compares each residual (`d` minus the expected value of
  the posterior divided by `sigma`) to the RMS of residuals. Data with
  residuals greater than `tol` times the RMS are identified as outliers. This
  process is then repeated using the subset of `d` which were not flagged as
  outliers. If no new outliers are detected in an iteration then the algorithm
  stops.

  Parameters
  ----------
  d : (N,) float array
    Observations

  s : (N,) float array
    One standard deviation uncertainty on the observations.

  mu : (N,) float array, optional
    Mean of the Gaussian process at the observation points. Defaults to zeros.

  sigma : (N,) array, (N, N) array, or (N, N) scipy sparse matrix, optional
    Covariance of the Gaussian process at the observation points. Defaults to
    zeros.

  p : (N, P) float array, optional
    Basis vectors for the Gaussian process evaluated at the observation points.
    Defaults to an (N, 0) array.

  tol : float, optional
    Outlier tolerance. Smaller values make the algorithm more likely to
    identify outliers. A good value is 4.0 and this should not be set any lower
    than 2.0.

  maxitr : int, optional
    Maximum number of iterations.

  Returns
  -------
  out : (N,) bool array
    Array indicating which data are outliers

  '''
  d = as_array(d, dtype=float)
  assert_shape(d, (None,), 'd')
  # number of observations
  n = d.shape[0]

  s = as_array(s, dtype=float)
  assert_shape(s, (n,), 's')

  if mu is None:
    mu = np.zeros((n,), dtype=float)

  else:
    mu = as_array(mu, dtype=float)
    assert_shape(mu, (n,), 'mu')

  if sigma is None:
    sigma = sp.csc_matrix((n, n), dtype=float)

  else:
    sigma = _as_covariance(sigma)
    assert_shape(sigma, (n, n), 'sigma')

  if p is None:
    p = np.zeros((n, 0), dtype=float)

  else:
    p = as_array(p, dtype=float)
    assert_shape(p, (n, None), 'p')

  # number of basis functions
  m = p.shape[1]
  # total number of outlier detection iterations completed thus far
  itr = 0
  # boolean array indicating outliers
  out = np.zeros(n, dtype=bool)
  while True:
    LOGGER.debug(
      'Starting iteration %s of outlier detection routine' % (itr+1))
    # remove rows and cols where `out` is True
    sigma_i = sigma[:, ~out][~out, :]
    p_i = p[~out]
    mu_i = mu[~out]
    d_i = d[~out]
    s_i = s[~out]
    # add data covariance to GP covariance. If an array is added to a sparse
    # matrix then the output is a matrix. as_sparse_or_array coerces it back to
    # an array
    sigma_i = as_sparse_or_array(sigma_i + _as_covariance(s_i))
    Ksolver = PartitionedPosDefSolver(sigma_i, p_i)
    vec1, vec2 = Ksolver.solve(d_i - mu_i, np.zeros(m))

    # dereference everything that we no longer need
    del sigma_i, mu_i, p_i, d_i, s_i, Ksolver

    fit = mu + sigma[:, ~out].dot(vec1) + p.dot(vec2)
    # find new outliers
    res = np.abs(fit - d)/s
    rms = np.sqrt(np.mean(res[~out]**2))
    if np.all(out == (res > tol*rms)):
      break

    else:
      out = res > tol*rms
      itr += 1
      if itr == maxitr:
        warnings.warn('Reached the maximum number of iterations')
        break

  LOGGER.debug(
    'Detected %s outliers out of %s observations' % (sum(out), len(out)))

  return out


def _io_is_checked(fin):
  '''
  Decorator that indicates the function has the appropriate input and output
  and does not need to be wrapped with the io check functions.
  '''
  fin._io_is_checked = None
  return fin


def _is_null(fin):
  '''
  Decorator that indicates the mean function returns zeros, covariance function
  returns zeros, or the basis functions are an empty set. This is used to avoid
  unnecessarily adding arrays of zeros or appending empty arrays.
  '''
  fin._is_null = None
  return fin


@_is_null
@_io_is_checked
def zero_mean(x, diff):
  '''mean function that returns zeros'''
  return np.zeros((x.shape[0],), dtype=float)


@_is_null
@_io_is_checked
def zero_variance(x, diff):
  '''variance function that returns zeros'''
  return np.zeros((x.shape[0],), dtype=float)


@_is_null
@_io_is_checked
def zero_sparse_covariance(x1, x2, diff1, diff2):
  '''covariance function that returns sparse zeros'''
  return sp.csc_matrix((x1.shape[0], x2.shape[0]), dtype=float)


@_is_null
@_io_is_checked
def zero_dense_covariance(x1, x2, diff1, diff2):
  '''covariance function that returns dense zeros'''
  return np.zeros((x1.shape[0], x2.shape[0]), dtype=float)


@_is_null
@_io_is_checked
def empty_basis(x, diff):
  '''empty set of basis functions'''
  return np.zeros((x.shape[0], 0), dtype=float)


def _default_variance(covariance):
  '''Converts a covariance function to a variance function'''
  @_io_is_checked
  def variance(x, diff):
    cov = covariance(x, x, diff, diff)
    # cov may be a CSC sparse matrix or an array. Either way, it has a
    # diagonal method
    out = cov.diagonal()
    return out

  return variance


def _mean_io_check(fin):
  '''
  Decorator that ensures the mean function takes two positional arguments and
  returns an array with the appropriate shape.
  '''
  if hasattr(fin, '_io_is_checked'):
    return fin

  arg_count = get_arg_count(fin)

  @_io_is_checked
  def mean_checked(x, diff):
    if arg_count == 1:
      # `fin` only takes one argument and is assumed to not be differentiable
      if any(diff):
        raise ValueError(
          'The mean of the `GaussianProcess` is not differentiable')

      out = fin(x)

    else:
      # otherwise it is assumed that `fin` takes two arguments
      out = fin(x, diff)

    out = as_array(out)
    assert_shape(out, (x.shape[0],), 'mean_output')
    return out

  return mean_checked


def _variance_io_check(fin):
  '''
  Decorator that ensures the variance function takes two positional arguments
  and returns an array with the appropriate shape.
  '''
  if hasattr(fin, '_io_is_checked'):
    return fin

  arg_count = get_arg_count(fin)

  @_io_is_checked
  def variance_checked(x, diff):
    if arg_count == 1:
      # `fin` only takes one argument and is assumed to not be differentiable
      if any(diff):
        raise ValueError(
          'The variance of the `GaussianProcess` is not differentiable')

      out = fin(x)

    else:
      # otherwise it is assumed that `fin` takes two arguments
      out = fin(x, diff)

    out = as_array(out)
    assert_shape(out, (x.shape[0],), 'variance_output')
    return out

  return variance_checked


def _covariance_io_check(fin):
  '''
  Decorator that ensures the covariance function takes four positional
  arguments and returns either an array or csc sparse matrix with the
  appropriate shape.
  '''
  if hasattr(fin, '_io_is_checked'):
    return fin

  arg_count = get_arg_count(fin)

  @_io_is_checked
  def covariance_checked(x1, x2, diff1, diff2):
    if arg_count == 2:
      # `fin` only takes two argument and is assumed to not be differentiable
      if any(diff1) | any(diff2):
        raise ValueError(
          'The covariance of the `GaussianProcess` is not differentiable')

      out = fin(x1, x2)

    else:
      # otherwise it is assumed that `fin` takes four arguments
      out = fin(x1, x2, diff1, diff2)

    out = as_sparse_or_array(out)
    assert_shape(out, (x1.shape[0], x2.shape[0]), 'covariance_output')
    return out

  return covariance_checked


def _basis_io_check(fin):
  '''
  Decorator that ensures the basis function takes two positional arguments and
  returns an array with the appropriate shape
  '''
  if hasattr(fin, '_io_is_checked'):
    return fin

  arg_count = get_arg_count(fin)

  @_io_is_checked
  def basis_checked(x, diff):
    if arg_count == 1:
      # `fin` only takes one argument and is assumed to not be differentiable
      if any(diff):
        raise ValueError(
          'The basis functions for the `GaussianProcess` are not '
          'differentiable')

      out = fin(x)

    else:
      # otherwise it is assumed that `fin` takes two arguments
      out = fin(x, diff)

    out = as_array(out)
    assert_shape(out, (x.shape[0], None), 'basis_output')
    return out

  return basis_checked


def _add(gp1, gp2):
  '''
  Returns a `GaussianProcess` which is the sum of two `GaussianProcess`
  instances.
  '''
  if hasattr(gp2._mean, '_is_null'):
    mean = gp1._mean

  elif hasattr(gp1._mean, '_is_null'):
    mean = gp2._mean

  else:
    @_io_is_checked
    def mean(x, diff):
      out = gp1._mean(x, diff) + gp2._mean(x, diff)
      return out

  if hasattr(gp2._variance, '_is_null'):
    variance = gp1._variance

  elif hasattr(gp1._variance, '_is_null'):
    variance = gp2._variance

  else:
    @_io_is_checked
    def variance(x, diff):
      out = gp1._variance(x, diff) + gp2._variance(x, diff)
      return out

  if hasattr(gp2._covariance, '_is_null'):
    covariance = gp1._covariance

  elif hasattr(gp1._covariance, '_is_null'):
    covariance = gp2._covariance

  else:
    @_io_is_checked
    def covariance(x1, x2, diff1, diff2):
      out = as_sparse_or_array(gp1._covariance(x1, x2, diff1, diff2) +
                               gp2._covariance(x1, x2, diff1, diff2))
      return out

  if hasattr(gp2._basis, '_is_null'):
    basis = gp1._basis

  elif hasattr(gp1._basis, '_is_null'):
    basis = gp2._basis

  else:
    @_io_is_checked
    def basis(x, diff):
      out = np.hstack((gp1._basis(x, diff),
                       gp2._basis(x, diff)))
      return out

  dim = _combined_dim(gp1.dim, gp2.dim)
  out = GaussianProcess(mean, covariance, basis=basis, variance=variance, 
                        dim=dim)
  return out


def _subtract(gp1, gp2):
  '''
  Returns a `GaussianProcess` which is the difference of two `GaussianProcess`
  instances.
  '''
  if hasattr(gp2._mean, '_is_null'):
    mean = gp1._mean

  elif hasattr(gp1._mean, '_is_null'):
    @_io_is_checked
    def mean(x, diff):
      out = -gp2._mean(x, diff)
      return out

  else:
    @_io_is_checked
    def mean(x, diff):
      out = gp1._mean(x, diff) - gp2._mean(x, diff)
      return out

  if hasattr(gp2._variance, '_is_null'):
    variance = gp1._variance

  elif hasattr(gp1._variance, '_is_null'):
    variance = gp2._variance

  else:
    @_io_is_checked
    def variance(x, diff):
      out = gp1._variance(x, diff) + gp2._variance(x, diff)
      return out

  if hasattr(gp2._covariance, '_is_null'):
    covariance = gp1._covariance

  elif hasattr(gp1._covariance, '_is_null'):
    covariance = gp2._covariance

  else:
    @_io_is_checked
    def covariance(x1, x2, diff1, diff2):
      out = as_sparse_or_array(gp1._covariance(x1, x2, diff1, diff2) +
                               gp2._covariance(x1, x2, diff1, diff2))
      return out

  if hasattr(gp2._basis, '_is_null'):
    basis = gp1._basis

  elif hasattr(gp1._basis, '_is_null'):
    basis = gp2._basis

  else:
    @_io_is_checked
    def basis(x, diff):
      out = np.hstack((gp1._basis(x, diff),
                       gp2._basis(x, diff)))
      return out

  dim = _combined_dim(gp1.dim, gp2.dim)
  out = GaussianProcess(mean, covariance, basis=basis, variance=variance, 
                        dim=dim)
  return out


def _scale(gp, c):
  '''
  Returns a scaled `GaussianProcess`.
  '''
  if hasattr(gp._mean, '_is_null'):
    mean = gp._mean

  else:
    @_io_is_checked
    def mean(x, diff):
      out = c*gp._mean(x, diff)
      return out

  if hasattr(gp._variance, '_is_null'):
    variance = gp._variance

  else:
    @_io_is_checked
    def variance(x, diff):
      out = c**2*gp._variance(x, diff)
      return out

  if hasattr(gp._covariance, '_is_null'):
    covariance = gp._covariance

  else:
    @_io_is_checked
    def covariance(x1, x2, diff1, diff2):
      out = c**2*gp._covariance(x1, x2, diff1, diff2)
      return out

  out = GaussianProcess(mean, covariance, basis=gp._basis, variance=variance, 
                        dim=gp.dim)
  return out


def _differentiate(gp, d):
  '''
  Differentiates a `GaussianProcess`.
  '''
  if hasattr(gp._mean, '_is_null'):
    mean = gp._mean

  else:
    @_io_is_checked
    def mean(x, diff):
      out = gp._mean(x, diff + d)
      return out

  if hasattr(gp._variance, '_is_null'):
    variance = gp._variance

  else:
    @_io_is_checked
    def variance(x, diff):
      out = gp._variance(x, diff + d)
      return out

  if hasattr(gp._covariance, '_is_null'):
    covariance = gp._covariance

  else:
    @_io_is_checked
    def covariance(x1, x2, diff1, diff2):
      out = gp._covariance(x1, x2, diff1 + d, diff2 + d)
      return out

  if hasattr(gp._basis, '_is_null'):
    basis = gp._basis

  else:
    @_io_is_checked
    def basis(x, diff):
      out = gp._basis(x, diff + d)
      return out

  dim = d.shape[0]
  out = GaussianProcess(mean, covariance, basis=basis, variance=variance, 
                        dim=dim)
  return out


def _condition(gp, y, d, sigma, p, obs_diff, build_inverse):
  '''
  Returns a conditioned `GaussianProcess`.
  '''
  @MemoizeArrayInput
  def precompute():
    # do as many calculations as possible without yet knowning where the
    # interpolation points will be. This function is memoized so that I can
    # easily dereference the kernel inverse matrix with "clear_caches".

    # GP mean at the observation points
    mu_y = gp._mean(y, obs_diff)
    # GP covariance at the observation points
    C_y = gp._covariance(y, y, obs_diff, obs_diff)
    # GP basis functions at the observation points
    p_y = gp._basis(y, obs_diff)
    # Only if noise basis vectors exist, append them to the GP basis vectors
    if p.shape[1] != 0:
      p_y = np.hstack((p_y, p))
      
    # add data noise to the covariance matrix
    C_y = as_sparse_or_array(C_y + sigma)
    # Create a factorization for the kernel, for rapid solving
    K_y_solver = PartitionedPosDefSolver(C_y, p_y, build_inverse=build_inverse)
    # evaluate the right-most operations for computing the mean since they do
    # not require knowledge of the interpolation points, store the intermediate
    # results as vec1 and vec2
    r = d - mu_y
    z = np.zeros((p_y.shape[1],), dtype=float)
    vec1, vec2 = K_y_solver.solve(r, z)
    return K_y_solver, vec1, vec2

  @_io_is_checked
  def mean(x, diff):
    _, vec1, vec2 = precompute()
    mu_x = gp._mean(x, diff)
    C_xy = gp._covariance(x, y, diff, obs_diff)
    p_x = gp._basis(x, diff)
    # Only if noise basis vectors exist, pad the GP basis vectors with that
    # many zeros
    if p.shape[1] != 0:
      p_x_pad = np.zeros((p_x.shape[0], p.shape[1]), dtype=float)
      p_x = np.hstack((p_x, p_x_pad))

    out = mu_x + C_xy.dot(vec1) + p_x.dot(vec2)
    return out

  @_io_is_checked
  def covariance(x1, x2, diff1, diff2):
    K_y_solver, _, _ = precompute()
    C_x1x2 = gp._covariance(x1, x2, diff1, diff2)
    C_x1y = gp._covariance(x1, y, diff1, obs_diff)
    C_x2y = gp._covariance(x2, y, diff2, obs_diff)
    p_x1 = gp._basis(x1, diff1)
    p_x2 = gp._basis(x2, diff2)
    # Only if noise basis vectors exist, pad the GP basis vectors with that
    # many zeros
    if p.shape[1] != 0:
      p_x1_pad = np.zeros((p_x1.shape[0], p.shape[1]), dtype=float)
      p_x2_pad = np.zeros((p_x2.shape[0], p.shape[1]), dtype=float)
      p_x1 = np.hstack((p_x1, p_x1_pad))
      p_x2 = np.hstack((p_x2, p_x2_pad))

    mat1, mat2 = K_y_solver.solve(C_x2y.T, p_x2.T)
    out = C_x1x2 - C_x1y.dot(mat1) - p_x1.dot(mat2)
    return out

  @_io_is_checked
  def variance(x, diff):
    K_y_solver, _, _ = precompute()
    var_x = gp._variance(x, diff)
    C_xy = gp._covariance(x, y, diff, obs_diff)
    p_x = gp._basis(x, diff)
    # Only if noise basis vectors exist, pad the GP basis vectors with that
    # many zeros
    if p.shape[1] != 0:
      p_x_pad = np.zeros((p_x.shape[0], p.shape[1]), dtype=float)
      p_x = np.hstack((p_x, p_x_pad))

    mat1, mat2 = K_y_solver.solve(C_xy.T, p_x.T)
    # Efficiently get the diagonals of C_xy.dot(mat1) and p_x.dot(mat2)
    if sp.issparse(C_xy):
      diag1 = C_xy.multiply(mat1.T).sum(axis=1).A[:, 0]
    else:
      diag1 = np.einsum('ij, ji->i', C_xy, mat1)
 
    diag2 = np.einsum('ij, ji->i', p_x, mat2)
    out = var_x - diag1 - diag2
    return out

  dim = y.shape[1]
  out = GaussianProcess(mean, covariance, variance=variance, dim=dim)
  return out


class GaussianProcess(object):
  '''
  A `GaussianProcess` instance represents a stochastic process which is defined
  in terms of a mean function, a covariance function, and (optionally) a set of
  basis functions. This class is used to perform basic operations on Gaussian
  processes which include addition, subtraction, scaling, differentiation,
  sampling, and conditioning.

  Parameters
  ----------
  mean : function
    Function which returns either the mean of the Gaussian process at `x` or a
    specified derivative of the mean at `x`. This has the call signature

    `out = mean(x)`

    or

    `out = mean(x, diff)`

    `x` is an (N, D) array of positions. `diff` is a (D,) int array derivative
    specification (e.g. [0, 1] indicates to return the derivative with respect
    to the second spatial dimension). `out` must be an (N,) array. If this
    function only takes one argument then it is assumed to not be
    differentiable and the `differentiate` method for the `GaussianProcess`
    instance will return an error.

  covariance : function
    Function which returns either the covariance of the Gaussian process
    between points `x1` and `x2` or the covariance of the specified derivatives
    of the Gaussian process between points `x1` and `x2`. This has the call
    signature

    `out = covariance(x1, x2)`

    or

    `out = covariance(x1, x2, diff1, diff2)`

    `x1` and `x2` are (N, D) and (M, D) arrays of positions, respectively.
    `diff1` and `diff2` are (D,) int array derivative specifications. `out` can
    be an (N, M) array or scipy sparse matrix (csc format would be most
    efficient). If this function only takes two arguments, then it is assumed
    to not be differentiable and the `differentiate` method for the
    `GaussianProcess` instance will return an error.

  basis : function, optional
    Function which returns either the basis functions evaluated at `x` or the
    specified derivative of the basis functions evaluated at `x`. This has the
    call signature

    `out = basis(x)`

    or

    `out = basis(x, diff)`

    `x` is an (N, D) array of positions. `diff` is a (D,) int array derivative
    specification. `out` is an (N, P) array where each column corresponds to a
    basis function. By default, a `GaussianProcess` instance contains no basis
    functions. If this function only takes one argument, then it is assumed to
    not be differentiable and the `differentiate` method for the
    `GaussianProcess` instance will return an error.

  variance : function, optional
    A function that returns the variance of the Gaussian process or its
    derivative at `x`. The has the call signature

    `out = variance(x)`

    or

    `out = variance(x, diff)`

    If this function is provided, it should be a more efficient alternative to
    evaluating the covariance matrix at `(x, x)` and then taking the diagonals.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` instance. An error
    will be raised if method arguments have a conflicting number of spatial
    dimensions.

  Notes
  -----
  1. This class does not check whether the specified covariance function is
  positive definite, making it easy to construct an invalid `GaussianProcess`
  instance. For this reason, one may prefer to create a `GaussianProcess` with
  one of the constructor functions (e.g., `gpse` or `gppoly`).

  2. A `GaussianProcess` returned by `add`, `subtract`, `scale`,
  `differentiate`, and `condition` has `mean`, `covariance`, and `basis`
  function which calls the `mean`, `covariance`, and `basis` functions of its
  parents. Due to this recursive implementation, the number of generations of
  children is limited by the maximum recursion depth.

  Examples
  --------
  Create a `GaussianProcess` describing Brownian motion

  >>> import numpy as np
  >>> from rbf.gauss import GaussianProcess
  >>> def mean(x): return np.zeros(x.shape[0])
  >>> def cov(x1, x2): return np.minimum(x1[:, None, 0], x2[None, :, 0])
  >>> gp = GaussianProcess(mean, cov, dim=1) # Brownian motion is 1D


  '''
  def __init__(self, mean, covariance, basis=None, variance=None, dim=None):
    self._mean = _mean_io_check(mean)
    self._covariance = _covariance_io_check(covariance)

    if basis is None:
      basis = empty_basis

    self._basis = _basis_io_check(basis)

    if variance is None:
      variance = _default_variance(self._covariance)

    self._variance = _variance_io_check(variance)

    self.dim = dim

  def __call__(self, *args, **kwargs):
    '''
    equivalent to calling `meansd`
    '''
    return self.meansd(*args, **kwargs)

  def __add__(self, other):
    '''
    equivalent to calling `add`
    '''
    return self.add(other)

  def __sub__(self, other):
    '''
    equivalent to calling `subtract`
    '''
    return self.subtract(other)

  def __mul__(self, c):
    '''
    equivalent to calling `scale`
    '''
    return self.scale(c)

  def __rmul__(self, c):
    '''
    equivalent to calling `scale`
    '''
    return self.__mul__(c)

  def __or__(self, args):
    '''
    equivalent to calling `condition` with positional arguments `args`.
    '''
    return self.condition(*args)

  def add(self, other):
    '''
    Adds two `GaussianProcess` instances.

    Parameters
    ----------
    other : GuassianProcess

    Returns
    -------
    out : GaussianProcess

    '''
    out = _add(self, other)
    return out

  def subtract(self, other):
    '''
    Subtracts two `GaussianProcess` instances.

    Parameters
    ----------
    other : GuassianProcess

    Returns
    -------
    out : GaussianProcess

    '''
    out = _subtract(self, other)
    return out

  def scale(self, c):
    '''
    Scales a `GaussianProcess`.

    Parameters
    ----------
    c : float

    Returns
    -------
    out : GaussianProcess

    '''
    c = np.float64(c)
    out = _scale(self, c)
    return out

  def differentiate(self, d):
    '''
    Returns the derivative of a `GaussianProcess`.

    Parameters
    ----------
    d : (D,) int array
      Derivative specification

    Returns
    -------
    out : GaussianProcess

    '''
    d = as_array(d, dtype=int)
    assert_shape(d, (self.dim,), 'd')

    out = _differentiate(self, d)
    return out

  def condition(self, y, d, sigma=None, p=None, obs_diff=None, 
                build_inverse=False):
    '''
    Returns a conditional `GaussianProcess` which incorporates the observed
    data, `d`.

    Parameters
    ----------
    y : (N, D) float array
      Observation points

    d : (N,) float array
      Observed values at `y`

    sigma : (N,) array, (N, N) array, or (N, N) scipy sparse matrix, optional
      Data uncertainty. If this is an (N,) array then it describes one standard
      deviation of the data error. If this is an (N, N) array then it describes
      the covariances of the data error. If nothing is provided then the error
      is assumed to be zero. Note that having zero uncertainty can result in
      numerically unstable calculations for large N.

    p : (N, P) array, optional
      Basis vectors for the noise. The data noise is assumed to contain some
      unknown linear combination of the columns of `p`.

    obs_diff : (D,) int array, optional
      Derivative of the observations. For example, use (1,) if the observations
      constrain the slope of a 1-D Gaussian process.
    
    build_inverse : bool, optional
      Whether to construct the inverse matrices rather than just the factors  

    Returns
    -------
    out : GaussianProcess

    '''
    ## Check the input for errors
    y = as_array(y, dtype=float)
    assert_shape(y, (None, self.dim), 'y')
		# number of observations and spatial dimensions
    n, dim = y.shape

    d = as_array(d, dtype=float)
    assert_shape(d, (n,), 'd')

    if sigma is None:
      sigma = sp.csc_matrix((n, n), dtype=float)

    else:
      sigma = _as_covariance(sigma)
      assert_shape(sigma, (n, n), 'sigma')

    if p is None:
      p = np.zeros((n, 0), dtype=float)

    else:
      p = as_array(p, dtype=float)
      assert_shape(p, (n, None), 'p')

    if obs_diff is None:
      obs_diff = np.zeros(dim, dtype=int)

    else:
      obs_diff = as_array(obs_diff, dtype=int)
      assert_shape(obs_diff, (dim,), 'obs_diff')

    out = _condition(self, y, d, sigma, p, obs_diff, 
                     build_inverse=build_inverse)
    return out

  def likelihood(self, y, d, sigma=None, p=None):
    '''
    Returns the log likelihood of drawing the observations `d` from this
    `GaussianProcess`. The observations could potentially have noise which is
    described by `sigma` and `p`. If the Gaussian process contains any basis
    functions or if `p` is specified, then the restricted likelihood is
    returned. For more information, see the documentation for
    `rbf.gauss.likelihood` and references therein.

    Parameters
    ----------
    y : (N, D) array
      Observation points.

    d : (N,) array
      Observed values at `y`.

    sigma : (N,) array, (N, N) array, or (N, N) sparse matrix, optional
      Data uncertainty. If this is an (N,) array then it describes one standard
      deviation of the data error. If this is an (N, N) array then it describes
      the covariances of the data error. If nothing is provided then the error
      is assumed to be zero. Note that having zero uncertainty can result in
      numerically unstable calculations for large N.

    p : (N, P) float array, optional
      Basis vectors for the noise. The data noise is assumed to contain some
      unknown linear combination of the columns of `p`.

    Returns
    -------
    out : float
      log likelihood.

    '''
    y = as_array(y, dtype=float)
    assert_shape(y, (None, self.dim), 'y')
    n, dim = y.shape # number of observations and dimensions

    d = as_array(d, dtype=float)
    assert_shape(d, (n,), 'd')

    if sigma is None:
      sigma = sp.csc_matrix((n, n), dtype=float)

    else:
      sigma = _as_covariance(sigma)
      assert_shape(sigma, (n, n), 'sigma')

    if p is None:
      p = np.zeros((n, 0), dtype=float)

    else:
      p = as_array(p, dtype=float)
      assert_shape(p, (n, None), 'p')

    obs_diff = np.zeros(dim, dtype=int)

    # find the mean, covariance, and basis for the combination of the Gaussian
    # process and the noise.
    mu = self._mean(y, obs_diff)

    gp_sigma = self._covariance(y, y, obs_diff, obs_diff)
    sigma = as_sparse_or_array(gp_sigma + sigma)

    gp_p = self._basis(y, obs_diff)
    p = np.hstack((gp_p, p))

    out = likelihood(d, mu, sigma, p=p)
    return out

  def outliers(self, y, d, sigma, tol=4.0, maxitr=50):
    '''
    Uses a data editing algorithm to identify outliers in `d`. Outliers are
    considered to be the data that are abnormally inconsistent with the
    `GaussianProcess`. This method can only be used for data that has nonzero,
    uncorrelated noise.

    The data editing algorithm first conditions the `GaussianProcess` with the
    observations, then it compares each residual (`d` minus the expected value
    of the posterior divided by `sigma`) to the RMS of residuals. Data with
    residuals greater than `tol` times the RMS are identified as outliers. This
    process is then repeated using the subset of `d` which were not flagged as
    outliers. If no new outliers are detected in an iteration then the
    algorithms stops.

    Parameters
    ----------
    y : (N, D) float array
      Observation points.

    d : (N,) float array
      Observed values at `y`

    sigma : (N,) float array
      One standard deviation uncertainty on `d`

    tol : float
      Outlier tolerance. Smaller values make the algorithm more likely to
      identify outliers. A good value is 4.0 and this should not be set any
      lower than 2.0.

    Returns
    -------
    out : (N,) bool array
      Boolean array indicating which data are outliers

    '''
    y = as_array(y, dtype=float)
    assert_shape(y, (None, self.dim), 'y')
    n, dim = y.shape # number of observations and dimensions

    d = as_array(d, dtype=float)
    assert_shape(d, (n,), 'd')

    # sigma is kept as a 1-D array
    sigma = as_array(sigma, dtype=float)
    assert_shape(sigma, (n,), 'sigma')

    obs_diff = np.zeros(dim, dtype=int)

    # find the mean, covariance, and basis for the combination of the Gaussian
    # process and the noise.
    gp_mu = self._mean(y, obs_diff)
    gp_sigma = self._covariance(y, y, obs_diff, obs_diff)
    gp_p = self._basis(y, obs_diff)
    out = outliers(d, sigma,
                   mu=gp_mu, sigma=gp_sigma,
                   p=gp_p, tol=tol, maxitr=maxitr)
    return out

  def basis(self, x, diff=None):
    '''
    Returns the basis functions evaluated at `x`.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points

    diff : (D,) int array
      Derivative specification

    Returns
    -------
    out : (N, P) array

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')

    if diff is None:
      diff = np.zeros(x.shape[1], dtype=int)

    else:
      diff = as_array(diff, dtype=int)
      assert_shape(diff, (x.shape[1],), 'diff')

    out = self._basis(x, diff)
    # return a dense copy of out
    out = as_array(out, copy=True)
    return out

  def mean(self, x, diff=None):
    '''
    Returns the mean of the proper component of the `GaussianProcess`.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points

    diff : (D,) int array
      Derivative specification

    Returns
    -------
    out : (N,) array

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')

    if diff is None:
      diff = np.zeros(x.shape[1], dtype=int)

    else:
      diff = as_array(diff, dtype=int)
      assert_shape(diff, (x.shape[1],), 'diff')

    out = self._mean(x, diff)
    # return a dense copy of out
    out = as_array(out, copy=True)
    return out

  def variance(self, x, diff=None):
    '''
    Returns the variance of the proper component of the `GaussianProcess`.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points

    diff : (D,) int array
      Derivative specification

    Returns
    -------
    out : (N,) array

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')

    if diff is None:
      diff = np.zeros(x.shape[1], dtype=int)

    else:
      diff = as_array(diff, dtype=int)
      assert_shape(diff, (x.shape[1],), 'diff')

    out = self._variance(x, diff)
    # return a dense copy of out
    out = as_array(out, copy=True)
    return out

  def covariance(self, x1, x2, diff1=None, diff2=None):
    '''
    Returns the covariance of the proper component of the `GaussianProcess`.

    Parameters
    ----------
    x1, x2 : (N, D) array
      Evaluation points

    diff1, diff2 : (D,) int array
      Derivative specification. For example, if `diff1` is (0,) and `diff2` is
      (1,), then the returned covariance matrix will indicate how the Gaussian
      process at `x1` covaries with the derivative of the Gaussian process at
      `x2`.

    Returns
    -------
    out : (N, N) array

    '''
    x1 = as_array(x1, dtype=float)
    assert_shape(x1, (None, self.dim), 'x1')

    x2 = as_array(x2, dtype=float)
    assert_shape(x2, (None, self.dim), 'x2')

    if diff1 is None:
      diff1 = np.zeros(x1.shape[1], dtype=int)

    else:
      diff1 = as_array(diff1, dtype=int)
      assert_shape(diff1, (x1.shape[1],), 'diff1')

    if diff2 is None:
      diff2 = np.zeros(x2.shape[1], dtype=int)

    else:
      diff2 = as_array(diff2, dtype=int)
      assert_shape(diff2, (x1.shape[1],), 'diff2')

    out = self._covariance(x1, x2, diff1, diff2)
    # return a dense copy of out
    out = as_array(out, copy=True)
    return out

  def meansd(self, x, chunk_size=100):
    '''
    Returns the mean and standard deviation of the proper component of the
    `GaussianProcess`. This does not return the full covariance matrix, making
    it appropriate for evaluating the `GaussianProcess` at many points.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points

    chunk_size : int, optional
      Break `x` into chunks with this size and evaluate the `GaussianProcess`
      for each chunk. This argument affects the speed and memory usage of this
      method, but it does not affect the output. Setting this to a larger value
      will reduce the number of python function call at the expense of
      increased memory usage.

    Returns
    -------
    out_mean : (N,) array
      Mean at `x`

    out_sd : (N,) array
      One standard deviation at `x`

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')
    # derivative of output will be zero
    diff = np.zeros(x.shape[1], dtype=int)

    # count is the total number of points evaluated thus far
    count = 0
    xlen = x.shape[0]
    out_mean = np.zeros(xlen, dtype=float)
    out_sd = np.zeros(xlen, dtype=float)
    # This block should run at least once to catch any potential errors
    while True:
      # only log the progress if the mean and sd are being build in multiple
      # chunks
      if xlen > chunk_size:
        LOGGER.debug(
          'Computing the mean and std. dev. (chunk size = %s) : '
          '%5.1f%% complete' % (chunk_size, (100.0*count)/xlen))

      start, stop = count, min(count+chunk_size, xlen)
      out_mean[start:stop] = self._mean(x[start:stop], diff)
      out_sd[start:stop] = np.sqrt(self._variance(x[start:stop], diff))
      count = stop
      if count == xlen:
        # break out of loop if all the points have been evaluated
        break

    if xlen > chunk_size:
      LOGGER.debug(
        'Computing the mean and std. dev. (chunk size = %s) : '
        '100.0%% complete' % chunk_size)

    return out_mean, out_sd

  def sample(self, x, c=None, use_cholesky=False, count=None):
    '''
    Draws a random sample from the `GaussianProcess`.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points.

    c : (P,) array, optional
      Coefficients for the basis functions. If this is not specified then they
      are set to zero.

    use_cholesky : bool, optional
      Indicates whether to use the Cholesky decomposition to create the sample.
      The Cholesky decomposition is faster but it assumes that the covariance
      matrix is numerically positive definite (i.e. there are no slightly
      negative eigenvalues due to rounding error).

    count : int, optional
      If given, `count` samples will be drawn

    Returns
    -------
    out : (N,) array

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')
    # derivative of the sample will be zero
    diff = np.zeros(x.shape[1], dtype=int)

    mu = self._mean(x, diff)
    sigma = self._covariance(x, x, diff, diff)
    p = self._basis(x, diff)

    if c is not None:
      c = as_array(c, dtype=float)

    else:
      c = np.zeros(p.shape[1])

    assert_shape(c, (p.shape[1],), 'c')
    out = _sample(mu, sigma, use_cholesky=use_cholesky, count=count) + p.dot(c)
    return out

  def is_positive_definite(self, x):
    '''
    Tests if the covariance matrix, which is the covariance function evaluated
    at `x`, is positive definite. This is done by testing if the Cholesky
    decomposition of the covariance matrix finishes successfully.

    Parameters
    ----------
    x : (N, D) array
      Evaluation points

    Returns
    -------
    out : bool

    Notes
    -----
    1. This function may return `False` even if the covariance function is
    positive definite. This is because some of the eigenvalues for the matrix
    are so small that they become slightly negative due to numerical rounding
    error. This is most notably the case for the squared exponential covariance
    function.

    '''
    x = as_array(x, dtype=float)
    assert_shape(x, (None, self.dim), 'x')
    diff = np.zeros(x.shape[1], dtype=int)

    cov = self._covariance(x, x, diff, diff)
    out = is_positive_definite(cov)
    return out

  def memoize(self):
    '''
    Memoizes the `_mean`, `_covariance`, and `_basis` methods for this
    `GaussianProcess`. This can improve performance by cutting out redundant
    computations, but it may also increase memory consumption.
    '''
    self._mean = MemoizeArrayInput(self._mean)
    self._covariance = MemoizeArrayInput(self._covariance)
    self._variance = MemoizeArrayInput(self._variance)
    self._basis = MemoizeArrayInput(self._basis)


def gpiso(phi, params, dim=None, check_finite=True):
  '''
  Creates an isotropic `GaussianProcess` instance which has a constant mean and
  a covariance function that is described by a radial basis function.

  Parameters
  ----------
  phi : str or RBF instance
    Radial basis function describing the covariance function. For example, use
    `rbf.basis.se` for a squared exponential covariance function. This must be
    positive definite.

  params : 3-tuple
    Tuple containing the mean, the variance, and the shape parameter for the
    Gaussian process, respectively.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` domain. An error will
    be raised if method arguments have a conflicting number of spatial
    dimensions.

  check_finite : bool, optional
    Indicates whether to check if the output for `phi` is finite. NaNs or Infs
    may be encountered if the `RBF` instance is not sufficiently
    differentiable.

  Returns
  -------
  out : GaussianProcess

  Notes
  -----
  Not all radial basis functions are positive definite, which means that it is
  possible to instantiate an invalid `GaussianProcess`. The method
  `is_positive_definite` provides a necessary but not sufficient test for
  positive definiteness. Examples of predefined `RBF` instances which are
  positive definite include: `rbf.basis.se`, `rbf.basis.ga`, `rbf.basis.exp`,
  `rbf.basis.iq`, `rbf.basis.imq`.

  '''
  phi = rbf.basis.get_rbf(phi)
  params = as_array(params, dtype=float)

  @_io_is_checked
  def mean(x, diff):
    a, b, c = params
    if not any(diff):
      out = np.full(x.shape[0], a, dtype=float)

    else:
      out = np.zeros(x.shape[0], dtype=float)

    return out

  @_io_is_checked
  def covariance(x1, x2, diff1, diff2):
    a, b, c = params
    diff = diff1 + diff2
    coeff = b*(-1)**sum(diff2)
    out = coeff*phi(x1, x2, eps=c, diff=diff)
    if check_finite:
      if not _all_is_finite(out):
        raise ValueError(
          'Encountered a non-finite RBF covariance. This may be because the '
          'basis function is not sufficiently differentiable.')

    return out

  @_io_is_checked
  def variance(x, diff):
    a, b, c = params
    coeff = b*(-1)**sum(diff)
    value = coeff*phi.center_value(eps=c, diff=2*diff)
    if check_finite:
      if not _all_is_finite(value):
        raise ValueError(
          'Encountered a non-finite RBF variance. This may be because the '
          'basis function is not sufficiently differentiable.')

    out = np.full(x.shape[0], value)
    return out
    
  out = GaussianProcess(mean, covariance, variance=variance, dim=dim)
  return out


def gpse(params, dim=None):
  '''
  Creates an isotropic `GaussianProcess` with a squared exponential covariance
  function.

  Parameters
  ----------
  params : 3-tuple
    Tuple containing the mean, the variance, and the shape parameter for the
    Gaussian process, respectively.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` domain. An error will
    be raised if method arguments have a conflicting number of spatial
    dimensions.

  Returns
  -------
  out : GaussianProcess

  Notes
  -----
  1. Some of the eigenvalues for squared exponential covariance matrices are
  very small and may be slightly negative due to numerical rounding error.
  Consequently, the Cholesky decomposition for a squared exponential covariance
  matrix will often fail. This becomes a problem when conditioning a squared
  exponential `GaussianProcess` with noise-free data.

  '''
  out = gpiso(rbf.basis.se, params, dim=dim, check_finite=False)
  return out


def gpexp(params, dim=None, check_finite=True):
  '''
  Creates an isotropic `GaussianProcess` with an exponential covariance
  function.

  Parameters
  ----------
  params : 3-tuple
    Tuple containing the mean, the variance, and the shape parameter for the
    Gaussian process, respectively.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` domain. An error will
    be raised if method arguments have a conflicting number of spatial
    dimensions.

  Returns
  -------
  out : GaussianProcess

  '''
  out = gpiso(rbf.basis.exp, params, dim=dim, check_finite=check_finite)
  return out


def gpbasis(basis, dim=None, dense=False):
  '''
  Creates an `GaussianProcess` consisting only of basis functions.

  Parameters
  ----------
  basis : function
    Function that takes either one argument, `x`, or two arguments, `x` and
    `diff`. `x` is an (N, D) array of positions and `diff` is a (D,) array
    specifying the derivative. This function returns an (N, P) array, where
    each column is a basis function evaluated at `x`.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` domain. An error will
    be raised if method arguments have a conflicting number of spatial
    dimensions.

  dense : bool, optional
    If True, then the covariance function returns a dense, rather than sparse,
    array of zeros. This is useful when the covariance matrices are relatively
    small and we do not want to incur the overhead of sparse matrices.

  Returns
  -------
  out : GaussianProcess

  '''
  if dense:
    out = GaussianProcess(zero_mean, zero_dense_covariance, basis=basis, 
                          variance=zero_variance, dim=dim)
  else:
    out = GaussianProcess(zero_mean, zero_sparse_covariance, basis=basis, 
                          variance=zero_variance, dim=dim)

  return out


def gppoly(order, dim=None, dense=False):
  '''
  Returns a `GaussianProcess` consisting of monomial basis functions. The
  monomials span the space of all polynomials with a user-specified order. If
  `order` = 0, then the basis functions consists of a constant term, if `order`
  = 1 then the basis functions consists of a constant and linear term, etc.

  Parameters
  ----------
  order : int
    Order of the basis functions.

  dim : int, optional
    Fixes the spatial dimensions of the `GaussianProcess` domain. An error will
    be raised if method arguments have a conflicting number of spatial
    dimensions.

  dense : bool, optional
    If True, then the covariance function returns a dense, rather than sparse,
    array of zeros. This is useful when the covariance matrices are relatively
    small and we do not want to incur the overhead of sparse matrices.

  Returns
  -------
  out : GaussianProcess

  '''
  @_io_is_checked
  def basis(x, diff):
    powers = rbf.poly.monomial_powers(order, x.shape[1])
    out = rbf.poly.mvmonos(x, powers, diff)
    return out

  out = gpbasis(basis, dim=dim, dense=dense)
  return out


def gpgibbs(ls, sigma, delta=1e-4):
  '''
  Returns a `GaussianProcess` with zero mean and a Gibbs covariance function.
  The Gibbs kernel has a spatially varying lengthscale.

  Parameters
  ----------
  ls: function
    Function that takes an (N, D) array of positions and returns an (N, D)
    array indicating the lengthscale along each dimension at those positions.

  sigma: float
    Standard deviation of the Gaussian process.

  delta: float, optional
    Finite difference spacing to use when calculating the derivative of the
    `GaussianProcess`. An analytical solution for the derivative is not
    available because the derivative of the `ls` function is unknown.

  '''
  @_io_is_checked
  @covariance_differentiator(delta)
  def covariance(x1, x2):
    '''
    covariance function for the Gibbs Gaussian process.
    '''
    dim = x1.shape[1]
    lsx1 = ls(x1)
    lsx2 = ls(x2)

    # sanitize the output for `ls`
    lsx1 = as_array(lsx1, dtype=float)
    lsx2 = as_array(lsx2, dtype=float)
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

  #TODO define the variance function    
  out = GaussianProcess(zero_mean, covariance)
  return out
