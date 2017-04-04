''' 
This module defines a class, *GaussianProcess*, which is an 
abstraction that allows one to easily work with Gaussian processes. 
One main use for the *GaussianProcess* class is Gaussian process 
regression (GPR). GPR is also known as Kriging or Least Squares 
Collocation.  It is a technique for constructing a continuous function 
from discrete observations by incorporating a stochastic prior model 
for the underlying function.  GPR is performed with the *condition* 
method of a *GaussianProcess* instance. In addition to GPR, the 
*GaussianProcess* class can be used for basic arithmetic with Gaussian 
processes and for generating random samples of a Gaussian process.

There are several existing python packages for Gaussian processes (See
www.gaussianprocess.org for an updated list of packages). This module
was written because existing software lacked the ability to 1) create
partially improper Gaussian processes 2) compute analytical
derivatives of a Gaussian process and 3) condition a Gaussian process
with derivative constraints. Other software packages have a strong
focus on optimizing hyperparameters based on data likelihood. This
module does not include any optimization routines and hyperparameters
are always explicitly specified by the user. However, the
*GaussianProcess* class contains the *likelihood* method which can be
used with functions from *scipy.optimize* to construct a
hyperparameter optimization routine.


Gaussian Processes
==================
A proper Gaussian process, :math:`u_o(x)`, is a stochastic function
that is normally distributed in the space of all functions. A Gaussian
process can be defined in terms of its mean, :math:`\\bar{u}(x)`, and
its covariance, :math:`C_u(x,x')`, as

.. math::
  u_o \\sim \\mathcal{N}\\left(\\bar{u},C_u\\right).

In this module, we adopt a more general definition which allows for
partially improper Gaussian processes (i.e. a Gaussian process which
has infinite variance along some directions).  We then consider a
Gaussian process, :math:`u(x)`, to be the combination of a proper
Gaussian process and a set of basis functions, :math:`\mathbf{p}_u(x)
= \{p_i(x)\}_{i=1}^m`, whose coefficients, :math:`\{c_i\}_{i=1}^m`,
have infinite variance. We express :math:`u(x)` as

.. math::
  u(x) = u_o(x) + \sum_{i=1}^m c_i p_i(x).

Throughout this module, we refer to :math:`\mathbf{p}_u(x)` as the
improper basis functions. We consider five operations on Gaussian
processes: addition, subtraction, scaling, differentiation, and
conditioning. Each operation produces another Gaussian process which
possesses the same five operations. These operations are described
below.


Operations on Gaussian Processes
================================

Addition
--------
Two uncorrelated Gaussian processes, :math:`u` and :math:`v`, can be 
added as

.. math::
  u(x) + v(x) = z(x)

where the mean, covariance, and improper basis functions for 
:math:`z` are

.. math::
  \\bar{z}(x) = \\bar{u}(x) + \\bar{v}(x),

.. math::
  C_z(x,x') = C_u(x,x') + C_v(x,x'),
  
and 

.. math::
  \mathbf{p}_z(x) = \mathbf{p}_u(x) \cup \mathbf{p}_v(x).

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
A Gaussian process can be differentiated along the direction
:math:`x_i` with the differential operator

.. math::
  D_x = \\frac{\partial}{\partial x_i}

as

.. math::
  D_xu(x) = z(x), 

where 

.. math::
  \\bar{z}(x) = D_x\\bar{u}(x),
  
.. math::
  C_z(x,x') = D_xD_{x'}C_u(x,x'),
  
and 

.. math::
  \mathbf{p}_z(x) = \\left\{D_x p_i(x) \mid p_i(x) \in 
                            \mathbf{p}_u(x)\\right\}


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
suitably padded with zeros. Note that there are no improper basis 
functions in :math:`z` because it is assumed that there is enough data
in :math:`\mathbf{d}` to constrain the basis functions in :math:`u`.
If :math:`\mathbf{d}` is not sufficiently informative then
:math:`\mathbf{K}(\mathbf{y})` will not be invertible. A necessary but 
not sufficient condition for :math:`\mathbf{K}(\mathbf{y})` to be 
invertible is that :math:`q \geq m`.


Special Classes of Gaussian Processes
=====================================

Isotropic Gaussian Processes
----------------------------
An isotropic Gaussian process has a constant mean and a covariance 
function that can be written as a function of :math:`||x - x'||_2`. We 
describe the mean and covariance for an isotropic Gaussian processes, 
:math:`u(x)`, as

.. math::
  \\bar{u}(x) = a,
  
and

.. math::
  C_u(x,x') = b\phi\\left(||x - x'||_2\ ; c\\right), 

Where :math:`\phi(r\ ; \epsilon)` is a positive definite radial basis 
function with shape parameter :math:`\epsilon`, and :math:`a`,
:math:`b`, and :math:`c` are distribution parameters. One common 
choice for :math:`\phi` is the squared exponential function,

.. math::
  \phi(r\ ;\epsilon) = \exp\\left(\\frac{-r^2}{\epsilon^2}\\right),

which has the useful property of being infinitely differentiable. An 
instance of an isotropic *GaussianProcess* can be created with the 
function *gpiso*.


Basis Function-Constrained Gaussian Processes
---------------------------------------------
A basis function-constrained Gaussian process, :math:`u(x)`, has
realizations that are limited to the space spanned by a set of
basis functions, :math:`\mathbf{f}(x) = \{f_i(x)\}_{i=1}^m`. That is
to say

.. math::
  u(x) = \sum_{i=1}^m a_i f_i(x),

where :math:`\mathbf{a} = \{a_i\}_{i=1}^m` and 

.. math::
  \mathbf{a} \\sim \mathcal{N}(\mathbf{\\bar{a}},\mathbf{C_a}). 
  
If the variance of :math:`\mathbf{a}` is infinite, then :math:`u(x)` 
can be viewed as a Gaussian process with zero mean, zero covariance, 
and :math:`\mathbf{f}(x)` are the improper basis functions. If
:math:`\mathbf{a}` has a finite variance, then the mean and covariance 
for :math:`u(x)` are described as

.. math::
  \\bar{u}(x) = \mathbf{f}(x)\mathbf{\\bar{a}}^T,
  
and

.. math::
  C_u(x,x') = \mathbf{f}(x)\mathbf{C_a}\mathbf{f}(x')^T.

The basis functions are commonly chosen to be the set of monomials 
that span the space of all polynomials with some degree, :math:`d`. 
For example, if :math:`x \in \mathbb{R}^2` and
:math:`d=1`, then the polynomial basis functions would be

.. math::
  \mathbf{f}(x) = \{1,x,y\}.

An instance of a basis function-constrained *GaussianProcess* can be
created with the functions *gpbfc* and *gpbfci*. If the basis
functions are monomials, then the the *GaussianProcess* can be created
with the function *gppoly*.


Instantiating a *GaussianProcesses*
===================================
This module provides multiple ways to instantiate a *GaussianProcess* 
instance. The most general way, but also the most error-prone, is to 
instantiate a *GaussianProcess* directly with its *__init__* method. 
This requires the user to supply a valid mean and covariance function.  
For example, the below code block demonstrates how to create a 
*GaussianProcess* representing Brownian motion.

>>> import numpy as np
>>> from rbf.gauss import GaussianProcess
>>> def mean(x): return np.zeros(x.shape[0]) 
>>> def cov(x1,x2): return np.min(np.meshgrid(x1[:,0],x2[:,0],indexing='ij'),axis=0)
>>> gp = GaussianProcess(mean,cov,dim=1) # Brownian motion is 1D

This module contains a helper function for generating isotropic
*GaussianProcess* instances, which is named *gpiso*. It requires the
user to specify a positive-definite *RBF* instance and the three
distribution parameters mentioned above. For example, the below code
generates a squared exponential Gaussian process with a=0.0, b=1.0,
and c=2.0.

>>> from rbf.basis import se
>>> from rbf.gauss import gpiso
>>> gp = gpiso(se,(0.0,1.0,2.0))

Since the squared exponential is so commonly used, this module 
contains the function *gpse* for generating a squared exponential 
*GaussianProcess* instances. The below code block produces a 
*GaussianProcess* that is equivalent to the one generated above.

>>> from rbf.gauss import gpse
>>> gp = gpse((0.0,1.0,2.0))

The function *gpbfci* is used for generating *GaussianProcess*
instances with improper basis functions. It requires the user to
specify a function which returns a set of basis functions evaluted at
*x*. For example,

>>> from rbf.gauss import gpbfc,gpbfci
>>> def basis(x): return np.array([np.sin(x[:,0]),np.cos(x[:,0])]).T
>>> gp = gpbfci(basis)

Use *gpbfc* to make a basis function-constrained *GaussianProcess*
which has coefficients with finite variance.

>>> mean = [0.0,0.0] # mean basis function coefficients
>>> sigma = [1.0,1.0] # uncertainty of basis function coefficients
>>> gp = gpbfc(basis,mean,sigma)

The function *gppoly* is a helper function for creating a Gaussian
process which has monomials for the improper basis functions. The
monomials span the space of all polynomials with some order. This just
requires the user to specify the polynomial order.

>>> from rbf.gauss import gppoly
>>> gp = gppoly(1)

Lastly, a *GaussianProcess* can be created by adding, subtracting, 
scaling, differentiating, or conditioning existing *GaussianProcess* 
instances. For example,

>>> gp  = gpse((0.0,1.0,2.0))
>>> gp += gppoly(2) # add second order polynomial basis 
>>> gp *= 2.0 # scale by 2
>>> gp  = gp.differentiate((2,)) # compute second derivative


References
==========
[1] Rasmussen, C., and Williams, C., Gaussian Processes for Machine 
Learning. The MIT Press, 2006.

'''
import numpy as np
import rbf.poly
import rbf.basis
import warnings
import rbf.mp
from collections import OrderedDict
import logging
import weakref
import inspect
from rbf.basis import _assert_shape
from scipy.linalg import solve_triangular 
from scipy.linalg import cholesky 
logger = logging.getLogger(__name__)


def _is_positive_definite(A):
  ''' 
  Tests if *A* is positive definite or nearly positive definite. If 
  any of the eigenvalues are significantly negative or complex then 
  this function return False. 
  '''
  vals = np.linalg.eigvals(A)
  # tolerance for negative or complex eigenvalues
  min_tol = max(np.linalg.norm(np.spacing(A)),np.sqrt(np.spacing(0)))
  tol = max(0.0001*vals.real.max(),min_tol)
  if np.any(vals.real < -tol):
    return False
  elif np.any(np.abs(vals.imag) > tol):
    return False
  else:
    return True


def _make_numerically_positive_definite(A):
  ''' 
  Iteratively adds a small value to the diagonal components of *A* 
  until it is numerically positive definite. This function is an 
  attempt to overcome numerical roundoff errors which cause the 
  eigenvalues of *A* to have slightly negative values when the true 
  eigenvalues of *A* should all be positive. It is assumed that *A* is 
  already nearly positive definite.   
  '''
  if not _is_positive_definite(A):
    raise np.linalg.LinAlgError(
      'The matrix *A* is not sufficiently close to being positive '
      'definite.')   
  
  itr = 0
  maxitr = 10
  n = A.shape[0]
  # minimum value to add to the diagonals. This is a bound on the 
  # eigenvalue error which results from rounding the values in *A* to 
  # the nearest float. Note that this does not take into account 
  # rounding error that has already accumulated in *A*. Thus the error 
  # is generally larger than this bound.
  min_eps = max(np.linalg.norm(np.spacing(A)),np.sqrt(np.spacing(0)))
  while True:
    vals = np.linalg.eigvalsh(A)
    if not np.any(vals <= 0.0):
      # exit successfully
      return
    
    if itr == maxitr:
      # exit with error
      raise np.linalg.LinAlgError(
        'Unable to make *A* positive definite after %s iterations.' 
        % maxitr)
      
    eps = max(-2*vals.min(),min_eps)
    A[range(n),range(n)] += eps
    itr += 1


def _cholesky(A,*args,**kwargs):
  ''' 
  Cholesky decomposition which is more forgiving of matrices that are 
  not numerically positive definite
  '''
  if A.shape == (0,0):
    return np.zeros((0,0))
    
  try:
    return cholesky(A,*args,**kwargs)
  except np.linalg.LinAlgError:  
    warnings.warn(
      'Failed to compute the Cholesky decomposition of *A*. This may '
      'be because *A* has slightly negative eigenvalues as a result of '
      'numerical rounding error. Small values will be added to the '
      'diagonals of *A* and the decomposition will be attempted again.')
    _make_numerically_positive_definite(A)   
    return cholesky(A,*args,**kwargs)


def _trisolve(G,d,*args,**kwargs):
  ''' 
  triangular solve which properly handles zero-length axes
  '''
  if any(i == 0 for i in d.shape):
    return np.zeros(d.shape)
  
  else:
    return solve_triangular(G,d,*args,**kwargs)  


def _sample(mean,cov):
  ''' 
  Draws a random sample from the Gaussian process with the specified 
  mean and covariance.
  '''   
  L = _cholesky(cov,lower=True,check_finite=False)
  w = np.random.normal(0.0,1.0,mean.shape[0])
  u = mean + L.dot(w)
  return u


class Memoize(object):
  ''' 
  Memoizing decorator. The output for calls to decorated functions 
  will be cached and reused if the function is called again with the 
  same arguments. The input arguments for decorated functions must all 
  be numpy arrays. Caches can be cleared with the module-level 
  function *clear_caches*. This is intendend to decorate the mean, 
  covariance, and basis functions for *GaussianProcess* instances. 
  '''
  # variable controlling the maximum cache size for all memoized 
  # functions
  MAX_CACHE_SIZE = 100
  # collection of weak references to all instances
  INSTANCES = []
  
  def __init__(self,fin):
    self.fin = fin
    self.cache = OrderedDict()
    Memoize.INSTANCES += [weakref.ref(self)]

  def __call__(self,*args):
    ''' 
    Calls the decorated function with *args* if the output is not 
    already stored in the cache. Otherwise, the cached value is 
    returned.
    '''
    key = tuple(a.tobytes() for a in args)
    if key not in self.cache:
      output = self.fin(*args)
      # make sure there is room for the new entry
      while len(self.cache) >= Memoize.MAX_CACHE_SIZE:
        self.cache.popitem(0)
        
      self.cache[key] = output
      
    return self.cache[key]

  def __repr__(self):
    return self.fin.__repr__()


def clear_caches():
  ''' 
  Dereferences the caches for all memoized functions. 
  '''
  for i in Memoize.INSTANCES:
    if i() is not None:
      # *i* will be done if it has no references. If references still 
      # exists, then give it a new empty cache.
      i().cache = OrderedDict()


def _add(gp1,gp2):
  '''   
  Returns a *GaussianProcess* which is the sum of two 
  *GaussianProcess* instances.
  '''
  def mean(x,diff):
    out = gp1._mean(x,diff) + gp2._mean(x,diff)
    return out       

  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out

  def basis(x,diff):
    out = np.hstack((gp1._basis(x,diff),
                     gp2._basis(x,diff)))
    return out                     
            
  dim = max(gp1.dim,gp2.dim)
  out = GaussianProcess(mean,covariance,basis=basis,dim=dim)
  return out
  

def _subtract(gp1,gp2):
  '''   
  Returns a *GaussianProcess* which is the difference of two 
  *GaussianProcess* instances.
  '''
  def mean(x,diff):
    out = gp1._mean(x,diff) - gp2._mean(x,diff)
    return out
      
  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out       
            
  def basis(x,diff):
    out = np.hstack((gp1._basis(x,diff),
                     gp2._basis(x,diff)))
    return out                     

  dim = max(gp1.dim,gp2.dim)
  out = GaussianProcess(mean,covariance,basis=basis,dim=dim)
  return out


def _scale(gp,c):
  '''   
  Returns a scaled *GaussianProcess*.
  '''
  def mean(x,diff):
    out = c*gp._mean(x,diff)
    return out

  def covariance(x1,x2,diff1,diff2):
    out = c**2*gp._covariance(x1,x2,diff1,diff2)
    return out
      
  # the basis functions are unchanged by scaling
  out = GaussianProcess(mean,covariance,basis=gp._basis,dim=gp.dim)
  return out


def _differentiate(gp,d):
  '''   
  Differentiates a *GaussianProcess*.
  '''
  def mean(x,diff):
    out = gp._mean(x,diff + d)
    return out 

  def covariance(x1,x2,diff1,diff2):
    out = gp._covariance(x1,x2,diff1+d,diff2+d)
    return out
      
  def basis(x,diff):
    out = gp._basis(x,diff + d)
    return out 
    
  dim = d.shape[0]
  out = GaussianProcess(mean,covariance,basis=basis,dim=dim)
  return out


def _condition(gp,y,d,sigma_noise,p_noise,obs_diff):
  '''   
  Returns a conditioned *GaussianProcess*.
  '''
  @Memoize
  def precompute():
    ''' 
    do as many calculations as possible without yet knowning where the 
    interpolation points will be.
    '''
    # compute K_y_inv
    Cu_yy = gp._covariance(y,y,obs_diff,obs_diff)
    p_y   = gp._basis(y,obs_diff)
    q,m,v = d.shape[0],p_y.shape[1],p_noise.shape[1]
    p_y = np.hstack((p_y,p_noise)) 
    K_y = np.zeros((q+m+v,q+m+v))
    K_y[:q,:q] = Cu_yy + sigma_noise
    K_y[:q,q:] = p_y
    K_y[q:,:q] = p_y.T
    try:
      K_y_inv = np.linalg.inv(K_y)[:q+m,:q+m]
      
    except np.linalg.LinAlgError:
      raise np.linalg.LinAlgError(
        'Failed to compute the inverse of K. This could be because '
        'the data is not able to constrain the basis functions. This '
        'error could also be caused by noise-free observations that '
        'are inconsistent with the Gaussian process.')

    # compute r
    r = np.zeros(q+m)
    r[:q] = d - gp._mean(y,obs_diff)
    return K_y_inv,r
    
  def mean(x,diff):
    K_y_inv,r = precompute()
    Cu_xy = gp._covariance(x,y,diff,obs_diff)
    p_x   = gp._basis(x,diff)
    k_xy  = np.hstack((Cu_xy,p_x))
    out   = gp._mean(x,diff) + k_xy.dot(K_y_inv.dot(r))
    return out

  def covariance(x1,x2,diff1,diff2):
    K_y_inv,r = precompute()
    Cu_x1x2 = gp._covariance(x1,x2,diff1,diff2)
    Cu_x1y  = gp._covariance(x1,y,diff1,obs_diff)
    Cu_x2y  = gp._covariance(x2,y,diff2,obs_diff)
    p_x1    = gp._basis(x1,diff1)
    p_x2    = gp._basis(x2,diff2)
    k_x1y   = np.hstack((Cu_x1y,p_x1))
    k_x2y   = np.hstack((Cu_x2y,p_x2))
    out = Cu_x1x2 - k_x1y.dot(K_y_inv).dot(k_x2y.T) 
    return out
  
  dim = y.shape[1]
  out = GaussianProcess(mean,covariance,dim=dim)
  return out


def _get_arg_count(func):
  ''' 
  Returns the number of function arguments. If this cannot be inferred 
  then -1 is returned.
  '''
  try:
    results = inspect.getargspec(func)
  except TypeError:
    return -1
      
  if (results.varargs is not None) | (results.keywords is not None):
    return -1

  else:
    return len(results.args)
  

def _zero_mean(x,diff):
  '''mean function that returns zeros'''
  return np.zeros((x.shape[0],),dtype=float)  


def _zero_covariance(x1,x2,diff1,diff2):
  '''covariance function that returns zeros'''
  return np.zeros((x1.shape[0],x2.shape[0]),dtype=float)  


def _empty_basis(x,diff):
  '''empty set of basis functions'''
  return np.zeros((x.shape[0],0),dtype=float)  
  

def likelihood(d,mu,sigma,p=None):
  ''' 
  Returns the log likelihood. If *p* is not specified, then the
  likelihood is the probability of observing *d* from a normally
  distributed random vector with mean *mu* and covariance *sigma*. If
  *d* is expected to contain some unknown linear combination of basis
  vectors (e.g. a constant offset or linear trend), then *p* should be
  specified with those basis vectors as its columns. When *p* is
  specified, the restricted likelihood is returned. The restricted
  likelihood is the probability of observing *R.dot(d)* from a
  normally distributed random vector with mean *R.dot(mu)* and
  covariance *R.dot(sigma).dot(R.T)*, where *R* is a matrix with rows
  that are orthogonal to the columns of *p*. In other words, if *p* is
  specified then the component of *d* which lies along the columns of
  *p* will be ignored.
  
  The restricted likelihood was first described by [1] and it is
  covered in more general reference books such as [2]. Both [1] and
  [2] are good sources for additional information.
  
  Parameters
  ----------
  d : (N,) array
    observations
  
  mu : (N,) array
    mean of the random vector
  
  sigma : (N,) or (N,N) array    
    If this is an (N,) array then it describes one standard deviation
    of the random vector. If this is an (N,N) array then it describes
    the covariances.
  
  p : (N,P) array, optional  
    Improper basis vectors. If specified, then *d* is assumed to
    contain some unknown linear combination of the columns of *p*.

  References
  ----------
  [1] Harville D. (1974). Bayesian Inference of Variance Components
  Using Only Error Contrasts. Biometrica.
  
  [2] Cressie N. (1993). Statistics for Spatial Data. John Wiley &
  Sons.
     
  '''
  d = np.asarray(d,dtype=float)
  _assert_shape(d,(None,),'d')

  mu = np.asarray(mu,dtype=float)
  _assert_shape(mu,(d.shape[0],),'mu')
  
  sigma = np.asarray(sigma,dtype=float) # data covariance
  if sigma.ndim == 1:
    # convert std. dev. to covariance
    sigma = np.diag(sigma**2)
    
  _assert_shape(sigma,(d.shape[0],d.shape[0]),'sigma')

  if p is None:
    p = np.zeros((d.shape[0],0))
  else:  
    p = np.asarray(p,dtype=float)
  
  _assert_shape(p,(d.shape[0],None),'p')
  
  n,m = p.shape
  A = _cholesky(sigma,lower=True)        # A.dot(A.T) = 
                                         #   sigma
                                         
  B = _trisolve(A,p,lower=True)          # B.T.dot(B) = 
                                         #   p.T.dot(inv(sigma)).dot(p)
                                         
  C = _cholesky(B.T.dot(B),lower=True)   # C.dot(C.T) = 
                                         #   p.T.dot(inv(sigma)).dot(p)
                                         
  D = _cholesky(p.T.dot(p),lower=True)   # D.dot(D.T) = 
                                         #   p.T.dot(p)
                                         
  a = _trisolve(A,d-mu,lower=True)       # a.T.dot(a) = 
                                         #   (d-mu).T.dot(inv(sigma)).dot(d-mu)
                                         
  b = _trisolve(C,B.T.dot(a),lower=True) # b.T.dot(b) = 
                                         #   (d-mu).T.dot(inv(sigma)).dot(p).dot(
                                         #   inv( p.T.dot(inv(sigma)).dot(p) )).dot(
                                         #   p.T.dot(inv(sigma)).dot(d - mu)   
  out = (np.sum(np.log(np.diag(D))) -
         np.sum(np.log(np.diag(A))) -
         np.sum(np.log(np.diag(C))) -
         0.5*a.T.dot(a) +
         0.5*b.T.dot(b) -
         0.5*(n-m)*np.log(2*np.pi))
  return out


class GaussianProcess(object):
  ''' 
  A *GaussianProcess* instance represents a stochastic process which
  is defined in terms of a mean function, a covariance function, and
  (optionally) a set of improper basis functions. This class is used
  to perform basic operations on Gaussian processes which include
  addition, subtraction, scaling, differentiation, sampling, and
  conditioning.
    
  Parameters
  ----------
  mean : function 
    Mean function for the Gaussian process. This takes either one 
    argument, *x*, or two arguments, *x* and *diff*. *x* is an (N,D) 
    array of positions and *diff* is a (D,) array specifying the 
    derivative. If the function only takes one argument, then the 
    function is assumed to not be differentiable. The function should 
    return an (N,) array.

  covariance : function
    Covariance function for the Gaussian process. This takes either
    two arguments, *x1* and *x2*, or four arguments, *x1*, *x2*,
    *diff1* and *diff2*. *x1* and *x2* are (N,D) and (M,D) arrays of
    positions, respectively. *diff1* and *diff2* are (D,) arrays
    specifying the derivatives with respect to *x1* and *x2*,
    respectively. If the function only takes two arguments, then the
    function is assumed to not be differentiable. The function should
    return an (N,M) array.

  basis : function, optional
    Improper basis functions. This function takes either one argument,
    *x*, or two arguments, *x* and *diff*. *x* is an (N,D) array of
    positions and *diff* is a (D,) array specifying the derivative.
    This function should return an (N,P) array, where each column is a
    basis function evaluated at *x*. By default, a *GaussianProcess*
    instance contains no improper basis functions.
        
  dim : int, optional  
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.
    
  Notes
  -----
  1. This class does not check whether the specified covariance 
  function is positive definite, making it easy construct an invalid 
  *GaussianProcess* instance. For this reason, one may prefer to 
  create a *GaussianProcess* with the functions *gpiso*, *gpbasis*, or 
  *gppoly*.
  
  2. A *GaussianProcess* returned by *add*, *subtract*, *scale*, 
  *differentiate*, and *condition* has *mean*, *covariance*, and 
  *basis* function which calls the *mean*, *covariance*, and *basis* 
  functions of its parents. Due to this recursive implementation, the 
  number of generations of children (for lack of a better term) is 
  limited by the maximum recursion depth.
  
  '''
  def __init__(self,mean,covariance,basis=None,dim=None):
    if _get_arg_count(mean) == 1:
      # if the mean function only takes one argument then make a 
      # wrapper for it which takes two arguments.
      def mean_with_diff(x,diff):
        if sum(diff) != 0: 
          raise ValueError(
            'The mean of the *GaussianProcess* is not differentiable')
          
        return mean(x)
    
      self._mean = mean_with_diff
    else:
      # otherwise, assume that the function can take two arguments
      self._mean = mean  
      
    if _get_arg_count(covariance) == 2:
      # if the covariance funciton only takes two argument then make a 
      # wrapper for it which takes four arguments.
      def covariance_with_diff(x1,x2,diff1,diff2):
        if (sum(diff1) != 0) | (sum(diff2) != 0): 
          raise ValueError(
            'The covariance of the *GaussianProcess* is not '
            'differentiable')
          
        return covariance(x1,x2)

      self._covariance = covariance_with_diff
    else:
      # otherwise, assume that the function can take four arguuments
      self._covariance = covariance
    
    if basis is None:  
      basis = _empty_basis
    
    if _get_arg_count(basis) == 1:
      # if the basis function only takes one argument then make a 
      # wrapper for it which takes two arguments.
      def basis_with_diff(x,diff):
        if sum(diff) != 0: 
          raise ValueError(
            'The improper basis functions for the *GaussianProcess* '
            'are not differentiable')
          
        return basis(x)
    
      self._basis = basis_with_diff
    else:
      # otherwise, assume that the function can take two arguments
      self._basis = basis
        
    self.dim = dim
  
  def __call__(self,*args,**kwargs):
    ''' 
    equivalent to calling *meansd*
    '''
    return self.meansd(*args,**kwargs)

  def __add__(self,other):
    ''' 
    equivalent to calling *add*
    '''
    return self.add(other)

  def __sub__(self,other):
    ''' 
    equivalent to calling *subtract*
    '''
    return self.subtract(other)

  def __mul__(self,c):
    ''' 
    equivalent to calling *scale*
    '''
    return self.scale(c)

  def __rmul__(self,c):
    ''' 
    equivalent to calling *scale*
    '''
    return self.__mul__(c)

  def __or__(self,args):
    ''' 
    equivalent to calling *condition* with positional arguments 
    *args*.
    '''
    return self.condition(*args)

  def add(self,other):
    ''' 
    Adds two *GaussianProcess* instances. 
    
    Parameters
    ----------
    other : GuassianProcess 
      
    Returns
    -------
    out : GaussianProcess 

    '''
    # make sure the dimensions of the GaussianProcess instances dont 
    # conflict
    if (self.dim is not None) & (other.dim is not None):
      if self.dim != other.dim:
        raise ValueError(
          'The number of spatial dimensions for the '
          '*GaussianProcess* instances are inconsitent.')
        
    out = _add(self,other)
    return out

  def subtract(self,other):
    '''  
    Subtracts two *GaussianProcess* instances.
    
    Parameters
    ----------
    other : GuassianProcess 
      
    Returns
    -------
    out : GaussianProcess 
      
    '''
    # make sure the dimensions of the GaussianProcess instances dont 
    # conflict
    if (self.dim is not None) & (other.dim is not None):
      if self.dim != other.dim:
        raise ValueError(
          'The number of spatial dimensions for the '
          '*GaussianProcess* instances are inconsitent.')

    out = _subtract(self,other)
    return out
    
  def scale(self,c):
    ''' 
    Scales a *GaussianProcess*.
    
    Parameters
    ----------
    c : float
      
    Returns
    -------
    out : GaussianProcess 
      
    '''
    c = np.float64(c)
    out = _scale(self,c)
    return out

  def differentiate(self,d):
    ''' 
    Returns the derivative of a *GaussianProcess*.
    
    Parameters
    ----------
    d : (D,) int array
      Derivative specification
      
    Returns
    -------
    out : GaussianProcess       

    '''
    d = np.asarray(d,dtype=int)
    _assert_shape(d,(self.dim,),'d')

    out = _differentiate(self,d)
    return out  

  def condition(self,y,d,sigma=None,p=None,obs_diff=None):
    ''' 
    Returns a conditional *GaussianProcess* which incorporates the 
    observed data, *d*.
    
    Parameters
    ----------
    y : (N,D) float array
      Observation points
    
    d : (N,) float array
      Observed values at *y*
      
    sigma : (N,) or (N,N) float array, optional
      Data uncertainty. If this is an (N,) array then it describes one 
      standard deviation of the data error. If this is an (N,N) array 
      then it describes the covariances of the data error. If nothing 
      is provided then the error is assumed to be zero. Note that 
      having zero uncertainty can result in numerically unstable 
      calculations for large N.

    p : (N,P) float array, optional  
      Improper basis vectors for the noise. The data noise is assumed
      to contain some unknown linear combination of the columns of
      *p*.

    obs_diff : (D,) int array, optional
      Derivative of the observations. For example, use (1,) if the 
      observations constrain the slope of a 1-D Gaussian process.
    
    Returns
    -------
    out : GaussianProcess
      
    '''
    ## Check the input for errors 
    y = np.asarray(y,dtype=float)
    _assert_shape(y,(None,self.dim),'y')
		# number of observations and spatial dimensions
    n,dim = y.shape 

    d = np.asarray(d,dtype=float)
    _assert_shape(d,(n,),'d')

    if sigma is None:
      sigma = np.zeros((n,n),dtype=float)      
    else:
      sigma = np.asarray(sigma,dtype=float)
      if sigma.ndim == 1:
        # convert standard deviations to covariances
        sigma = np.diag(sigma**2)

      _assert_shape(sigma,(n,n),'sigma')
        
    if p is None:
      p = np.zeros((n,0))
    else:
      p = np.asarray(p,dtype=float)
      _assert_shape(p,(n,None),'p')
      
    if obs_diff is None:
      obs_diff = np.zeros(dim,dtype=int)
    else:
      obs_diff = np.asarray(obs_diff,dtype=int)
      _assert_shape(obs_diff,(dim,),'obs_diff')
    
    out = _condition(self,y,d,sigma,p,obs_diff)
    return out

  def likelihood(self,y,d,sigma=None,p=None,obs_diff=None):
    ''' 
    Returns the log likelihood of drawing the observations *d* from
    this *GaussianProcess*. The observations could potentially have noise
    which is described by *sigma* and *p*. If the Gaussian process
    contains any improper basis functions or if *p* is specified, then the
    restricted likelihood is returned. For more information, see the
    documentation for *rbf.gauss.likelihood* and references therein.

    Parameters
    ----------
    y : (N,D) array
      Observation points.
    
    d : (N,) array
      Observed values at *y*.
      
    sigma : (N,) or (N,N) float array, optional
      Data uncertainty. If this is an (N,) array then it describes one
      standard deviation of the data error. If this is an (N,N) array
      then it describes the covariances of the data error. If nothing
      is provided then the error is assumed to be zero. Note that
      having zero uncertainty can result in numerically unstable
      calculations for large N.
   
    p : (N,P) float array, optional 
      Improper basis vectors for the noise. The data noise is assumed
      to contain some unknown linear combination of the columns of
      *p*. 
      
    obs_diff : (D,) tuple, optional
      Derivative of the observations. For example, use (1,) if the 
      observations constrain the slope of a 1-D Gaussian process.

    Returns
    -------
    out : float
      log likelihood.
      
    '''
    y = np.asarray(y,dtype=float)
    _assert_shape(y,(None,self.dim),'y')
    n,dim = y.shape # number of observations and dimensions

    d = np.asarray(d,dtype=float)
    _assert_shape(d,(n,),'d')

    if sigma is None:
      sigma = np.zeros((n,n),dtype=float)
    else:
      sigma = np.asarray(sigma,dtype=float)
      if sigma.ndim == 1:
        # If sigma is a 1d array then it contains std. dev. uncertainties.  
        # Convert sigma to a covariance matrix
        sigma = np.diag(sigma**2)

      _assert_shape(sigma,(n,n),'sigma')
    
    if p is None:
      p = np.zeros((n,0))
    else:
      p = np.asarray(p,dtype=float)
      _assert_shape(p,(n,None),'p')

    if obs_diff is None:
      obs_diff = np.zeros(dim,dtype=int)
    else:
      obs_diff = np.asarray(obs_diff,dtype=int)
      _assert_shape(obs_diff,(dim,),'obs_diff')

	  # find the mean, covariance, and improper basis for the combination
  	# of the Gaussian process and the noise.
    mu = self._mean(y,obs_diff)
    sigma = self._covariance(y,y,obs_diff,obs_diff) + sigma
    p = np.hstack((self._basis(y,obs_diff),p))
    out = likelihood(d,mu,sigma,p=p)
    return out

  def basis(self,x,diff=None):
    ''' 
    Returns the improper basis functions evaluated at *x*.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
        
    diff : (D,) tuple
      Derivative specification    
      
    Returns
    -------
    out : (N,P) array  

    '''
    x = np.asarray(x,dtype=float)
    _assert_shape(x,(None,self.dim),'x')
    
    if diff is None:  
      diff = np.zeros(x.shape[1],dtype=int)
    else:
      diff = np.asarray(diff,dtype=int)
      _assert_shape(diff,(x.shape[1],),'diff')
      
    out = self._basis(x,diff)
    out = np.array(out,copy=True)
    return out

  def mean(self,x,diff=None,retry=1):
    ''' 
    Returns the mean of the proper component of the *GaussianProcess*.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
        
    diff : (D,) tuple
      Derivative specification    
      
    retry : int, optional
      If the mean of the Gaussian process evaluates to a non-finite 
      value then this many attempts will be made to recompute it. This 
      option was added because my CPU is surprisingly unreliable when 
      using multiple cores and my data occassionally gets corrupted. 
      Hopefully, I can resolve my own computer problems and this 
      option will not be needed.
      
    Returns
    -------
    out : (N,) array  

    '''
    x = np.asarray(x,dtype=float)
    _assert_shape(x,(None,self.dim),'x')
    
    if diff is None:  
      diff = np.zeros(x.shape[1],dtype=int)
    else:
      diff = np.asarray(diff,dtype=int)
      _assert_shape(diff,(x.shape[1],),'diff')
      
    out = self._mean(x,diff)
    # If *out* is not finite then warn the user and attempt to compute 
    # it again. An error is raised after *retry* attempts.
    if not np.all(np.isfinite(out)):
      if retry > 0:
        warnings.warn(
          'Encountered non-finite value in the mean of the Gaussian '
          'process. This may be due to a CPU fluke. Memoized function ' 
          'caches will be cleared and another attempt will be made to '
          'compute the mean.')
        clear_caches()  
        return self.mean(x,diff=diff,retry=retry-1)
      else:    
        raise ValueError(
          'Encountered non-finite value in the mean of the Gaussian '
          'process.')     

    # return a copy of *out* that is safe to write to
    out = np.array(out,copy=True)
    return out

  def covariance(self,x1,x2,diff1=None,diff2=None,retry=1):
    ''' 
    Returns the covariance of the proper component of the
    *GaussianProcess*.
    
    Parameters
    ----------
    x1,x2 : (N,D) array
      Evaluation points
        
    diff1,diff2 : (D,) tuple
      Derivative specification. For example, if *diff1* is (0,) and 
      *diff2* is (1,), then the returned covariance matrix will indicate 
      how the Gaussian process at *x1* covaries with the derivative of 
      the Gaussian process at *x2*.

    retry : int, optional
      If the covariance of the Gaussian process evaluates to a 
      non-finite value then this many attempts will be made to 
      recompute it. This option was added because my CPU is 
      surprisingly unreliable when using multiple cores and my data 
      occassionally gets corrupted. Hopefully, I can resolve my own 
      computer problems and this option will not be needed.
      
    Returns
    -------
    out : (N,N) array    
    
    '''
    x1 = np.asarray(x1,dtype=float)
    _assert_shape(x1,(None,self.dim),'x1')

    x2 = np.asarray(x2,dtype=float)
    _assert_shape(x2,(None,self.dim),'x2')

    if diff1 is None:
      diff1 = np.zeros(x1.shape[1],dtype=int)
    else:
      diff1 = np.asarray(diff1,dtype=int)
      _assert_shape(diff1,(x1.shape[1],),'diff1')

    if diff2 is None:  
      diff2 = np.zeros(x2.shape[1],dtype=int)
    else:
      diff2 = np.asarray(diff2,dtype=int)
      _assert_shape(diff2,(x1.shape[1],),'diff2')
      
    out = self._covariance(x1,x2,diff1,diff2)
    # If *out* is not finite then warn the user and attempt to compute 
    # it again. An error is raised after *retry* attempts.
    if not np.all(np.isfinite(out)):
      if retry > 0:
        warnings.warn(
          'Encountered non-finite value in the covariance of the '
          'Gaussian process. This may be due to a CPU fluke. Memoized ' 
          'function caches will be cleared and another attempt will '
          'be made to compute the covariance.')
        clear_caches()  
        return self.covariance(x1,x2,diff1=diff1,diff2=diff2,
                               retry=retry-1)
      else:    
        raise ValueError(
          'Encountered non-finite value in the covariance of the '
          'Gaussian process.')     

    # return a copy of *out* that is safe to write to
    out = np.array(out,copy=True)
    return out
    
  def meansd(self,x,max_chunk=100):
    ''' 
    Returns the mean and standard deviation of the proper component of
    the *GaussianProcess*. This does not return the full covariance
    matrix, making it appropriate for evaluating the *GaussianProcess*
    at many points.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    max_chunk : int, optional  
      Break *x* into chunks with this size and evaluate the Gaussian 
      process for each chunk. This argument affects the speed and 
      memory usage of this method, but it does not affect the output. 
      Setting this to a larger value will reduce the number of python 
      function call at the expense of increased memory usage.
    
    Returns
    -------
    out_mean : (N,) array
      Mean of the stochastic component of the *GaussianProcess* at *x*.
    
    out_sd : (N,) array  
      One standard deviation uncertainty of the stochastic component 
      of the *GaussianProcess* at *x*.
      
    '''
    count = 0
    x = np.asarray(x,dtype=float)
    q = x.shape[0]
    out_mean = np.zeros(q,dtype=float)
    out_sd = np.zeros(q,dtype=float)
    while True:
      idx = range(count,min(count+max_chunk,q))
      out_mean[idx] = self.mean(x[idx])
      cov = self.covariance(x[idx],x[idx])
      var = np.diag(cov)
      out_sd[idx] = np.sqrt(var)
      count = min(count+max_chunk,q)
      if count == q:
        break
    
    return out_mean,out_sd

  def sample(self,x,c=None):  
    '''  
    Draws a random sample from the *GaussianProcess*.  
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points.
    
    c : (P,) array, optional
      Coefficients for the improper basis functions. If this is not
      specified then they are set to zero.
      
    Returns
    -------
    out : (N,) array      
    
    '''
    mu = self.mean(x)
    sigma = self.covariance(x,x)
    p = self.basis(x)
    if c is not None:
      c = np.asarray(c,dtype=float)
    else:
      c = np.zeros(p.shape[1])  
      
    _assert_shape(c,(p.shape[1],),'c')    
    out = _sample(mu,sigma) + p.dot(c)
    return out
    
  def is_positive_definite(self,x):
    '''     
    Tests if the covariance matrix, which is the covariance function
    evaluated at *x*, is positive definite. This is done by checking
    if all of its eigenvalues are positive. An affirmative result from
    this test is necessary but insufficient to ensure that the
    covariance function is positive definite.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points

    Returns
    -------
    out : bool

    '''
    cov = self.covariance(x,x)    
    out = _is_positive_definite(cov)
    return out  
    

def gpiso(phi,params,dim=None):
  ''' 
  Creates an isotropic *GaussianProcess* instance which has a constant 
  mean and a covariance function that is described by a radial basis 
  function.
  
  Parameters
  ----------
  phi : RBF instance
    Radial basis function describing the covariance function. For 
    example, use *rbf.basis.se* for a squared exponential covariance 
    function.

  params : 3-tuple  
    Tuple of three parameters, *a*, *b*, and *c*, describing the
    probability distribution. *a* is the mean, *b* scales the
    covariance function, and *c* is the shape parameter. When *phi* is
    set to *rbf.basis.se*, *b* and *c* describe the variance and the
    characteristic length-scale, respectively.
  
  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.
      
  Returns
  -------
  out : GaussianProcess

  Notes
  -----
  1. If *phi* is scale invariant, such as for odd order polyharmonic 
  splines, then *b* and *c* have inverse effects on the resulting 
  *GaussianProcess* and thus only one of them needs to be chosen while 
  the other can be fixed at an arbitary value.
  
  2. Not all radial basis functions are positive definite, which means 
  that it is possible to instantiate an invalid *GaussianProcess*. The 
  method *is_positive_definite* provides a necessary but not 
  sufficient test for positive definiteness. Examples of predefined 
  *RBF* instances which are positive definite include: *rbf.basis.se*, 
  *rbf.basis.ga*, *rbf.basis.exp*, *rbf.basis.iq*, *rbf.basis.imq*.

  '''
  params = np.asarray(params,dtype=float)  
  
  def mean(x,diff):
    a,b,c = params  
    if sum(diff) == 0:
      out = np.full(x.shape[0],a,dtype=float)
    else:
      out = np.zeros(x.shape[0],dtype=float)

    return out
      
  @Memoize
  def covariance(x1,x2,diff1,diff2):
    a,b,c = params  
    diff = diff1 + diff2
    out = b*(-1)**sum(diff2)*phi(x1,x2,eps=c,diff=diff)
    if not np.all(np.isfinite(out)):
      raise ValueError(
        'Encountered a non-finite RBF covariance. This may be '
        'because the basis function is not sufficiently '
        'differentiable.')

    return out

  out = GaussianProcess(mean,covariance,dim=dim)
  return out


def gpse(params,dim=None):
  ''' 
  Creates an isotropic *GaussianProcess* with a squared exponential 
  covariance function. 
  
  Parameters
  ----------
  params : 3-tuple  
    Tuple of three distribution parameters, *a*, *b*, and *c*. They
    describe the mean, variance, and the characteristic length-scale,
    respectively.
  
  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.
      
  Returns
  -------
  out : GaussianProcess
  '''
  out = gpiso(rbf.basis.se,params,dim=dim)
  return out


def gpexp(params,dim=None):
  ''' 
  Creates an isotropic *GaussianProcess* with an exponential 
  covariance function.
  
  Parameters
  ----------
  params : 3-tuple  
    Tuple of three distribution parameters, *a*, *b*, and *c*. They 
    describe the mean, variance, and the characteristic length-scale, 
    respectively.
  
  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.
      
  Returns
  -------
  out : GaussianProcess
  '''
  out = gpiso(rbf.basis.exp,params,dim=dim)
  return out


def gpbfc(basis,mu,sigma,dim=None):
  ''' 
  Creates a basis function-constrained *GaussianProcess* instance.
  Realizations of the *GaussianProcess* are linear combinations of the
  basis functions and the basis function coefficients have a
  distribution described by *mu* and *sigma*.
  
  Parameters
  ----------
  basis : function
    Function that takes either one argument, *x*, or two arguments, 
    *x* and *diff*. *x* is an (N,D) array of positions and *diff* is a 
    (D,) array specifying the derivative. This function returns an 
    (N,P) array, where each column is a basis function evaluated at 
    *x*. 
  
  mu : (P,) array
    Expected value of the basis function coefficients.
  
  sigma : (P,) or (P,P) array   
    If this is a (P,) array then it indicates the standard deviation 
    of the basis function coefficients. If it is a (P,P) array then it 
    indicates the covariances. 

  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.

  Returns
  -------
  out : GaussianProcess
    
  '''
  # make sure basis can take two arguments
  if _get_arg_count(basis) == 1:
    # if the basis function only takes one argument then make a 
    # wrapper for it which takes two arguments.
    def basis_with_diff(x,diff):
      if sum(diff) != 0: 
        raise ValueError(
          'The basis functions for the *GaussianProcess* instance '
          'are not differentiable.')
          
      return basis(x)
    
  else:
    # otherwise, assume that the function can take two arguments
    basis_with_diff = basis
      
  mu = np.asarray(mu,dtype=float)
  sigma = np.asarray(sigma,dtype=float)
  if sigma.ndim == 1:
      # if *sigma* is one dimensional then it contains standard 
      # deviations. These are converted to a covariance matrix.
      sigma = np.diag(sigma**2)
  
  def mean(x,diff):
    G = basis_with_diff(x,diff)
    out = G.dot(mu)
    return out
    
  def covariance(x1,x2,diff1,diff2):
    G1 = basis_with_diff(x1,diff1)
    G2 = basis_with_diff(x2,diff2)
    out = G1.dot(sigma).dot(G2.T)
    return out
    
  out = GaussianProcess(mean,covariance,dim=dim)
  return out  


def gpbfci(basis,dim=None):
  ''' 
  Creates an *GaussianProcess* consisting of improper basis functions.

  Parameters
  ----------
  basis : function
    Function that takes either one argument, *x*, or two arguments, 
    *x* and *diff*. *x* is an (N,D) array of positions and *diff* is a 
    (D,) array specifying the derivative. This function returns an 
    (N,P) array, where each column is a basis function evaluated at 
    *x*.

  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.

  Returns
  -------
  out : GaussianProcess
    
  '''
  out = GaussianProcess(_zero_mean,_zero_covariance,basis=basis,dim=dim)
  return out


def gppoly(order,dim=None):
  ''' 
  Returns a *GaussianProcess* consisting of monomial improper basis
  functions. The monomials span the space of all polynomials with a
  user-specified order. If *order* = 0, then the improper basis
  functions consists of a constant term, if *order* = 1 then the basis
  functions consists of a constant and linear term, etc.
  
  Parameters
  ----------
  order : int  
    Order of the basis functions.
    
  dim : int, optional
    Fixes the spatial dimensions of the *GaussianProcess* domain. An 
    error will be raised if method arguments have a conflicting number 
    of spatial dimensions.

  Returns
  -------
  out : GaussianProcess  
    
  '''
  @Memoize
  def basis(x,diff):
    powers = rbf.poly.powers(order,x.shape[1])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  out = gpbfci(basis,dim=dim)  
  return out
