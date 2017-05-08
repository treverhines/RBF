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
from collections import OrderedDict
import logging
import warnings
import weakref
import inspect
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rbf.basis import _assert_shape
from scipy.linalg.lapack import dpotri,dpotrf,dtrtrs
from code import InteractiveConsole
logger = logging.getLogger(__name__)

try:
  from sksparse import cholmod
  HAS_CHOLMOD = True
  
except ImportError:
  HAS_CHOLMOD = False
  logger.info(
    'Could not import CHOLMOD. Sparse matrices will be converted to '
    'dense for all Cholesky decompositions. To install CHOLMOD and its '
    'python wrapper, follow the instructions at '
    'https://scikit-sparse.readthedocs.io')
    

class NotPositiveDefiniteError(Exception):
  ''' 
  Error raised when Cholesky decompositions fails
  '''  
  pass


def _lapack_cholesky(A,**kwargs):  
  ''' 
  Computes the cholesky decomposition of *A* using LAPACK. kwargs are
  passed to *dpotrf*.
  '''
  if A.shape == (0,0):
    return np.zeros((0,0),dtype=float)
    
  L,info = dpotrf(A,**kwargs) 
  if info > 0:  
    raise NotPositiveDefiniteError(
      'The leading minor of order %s is not positive definite, and '
      'the factorization could not be completed. ' % info)

  elif info < 0:
    raise ValueError(
      'The %s-th argument has an illegal value.' % (-info))
  
  else:
    # finished successfully
    return L    


def _lapack_cholesky_inv(A):
  ''' 
  Returns the inverse of the positive definite matrix using LAPACK. It
  is assumed that *A* is a double precision numpy array.
  '''
  n,m = A.shape
  if (n,m) == (0,0):
    return np.zeros((0,0),dtype=float)

  L = _lapack_cholesky(A,lower=True)
  # Linv is the lower triangular components of the inverse.
  Linv,info = dpotri(L,lower=True)
  
  if info < 0:
    raise ValueError(
      'The %s-th argument had an illegal value.' % (-info))

  elif info > 0:
    raise np.linalg.LinAlgError(
      'The (%s,%s) element of the factor U or L is zero, and the '
      'inverse could not be computed.' % (info,info))

  else:
    # the decomposition exited successfully
    # reflect Linv over the diagonal      
    Linv = Linv + Linv.T
    # the diagonals are twice as big as they should be
    diag = Linv.diagonal()
    diag.flags.writeable = True
    diag *= 0.5
    return Linv


def _lapack_solve_triangular(G,d,**kwargs):
  ''' 
  Solve the triangular system of equations. *G* and *d* are both
  double precision numpy arrays. arguments get passed to the LAPACK
  function *dtrtrs*
  '''
  if any(i == 0 for i in d.shape):
    return np.zeros(d.shape)
  
  soln,info = dtrtrs(G,d,**kwargs)  
  if info < 0:
    raise ValueError(
      'The %s-th argument had an illegal value' % (-info))

  elif info > 0:
    raise np.linalg.LinAlgError(
      'The %s-th diagonal element of A is zero, indicating that '
      'the matrix is singular and the solutions X have not been '
      'computed.' % info)

  else:
    # finished successfully
    return soln


def _asarray(A,dtype=None,copy=False):
  ''' 
  If *A* is a scipy sparse matrix then return it as a csc matrix.
  Otherwise, return it as a numpy array.
  '''
  if sp.issparse(A):
    return sp.csc_matrix(A,dtype=dtype,copy=copy)
  
  else: 
    return np.array(A,dtype=dtype,copy=copy)  


def _all_is_finite(A):
  ''' 
  returns True if all values in *A* are finite. *A* can be a numpy
  array or a scipy sparse matrix.
  '''
  if sp.issparse(A):
    # get all the nonzero entries
    return np.all(np.isfinite(A.data))
    
  else:
    return np.all(np.isfinite(A))


def _cholesky(A):
  ''' 
  Returns the Cholesky decomposition of a sparsity-preserving
  permutation of *A*. See the documentation of sksparse.cholmod for
  clarification.
  
  Parameters
  ----------
  A : (N,N) numpy array or sparse matrix
  
  Returns
  -------
  L : (N,N) numpy array or sparse matrix
    cholesky decomposition of permuted *A*
    
  perm : (N,) int array
    permutation indices of *A*
    
  Notes
  -----
  To recover the matrix from the decompositon:

    >>> L,perm = _cholesky(A)
    >>> perm_inv = np.argsort(perm)
    >>> A_recovered = L[perm_inv,:].dot(L[perm_inv,:].T)
    
  '''
  n = A.shape[0]
  # if CHOLMOD is not available then make *A* dense if it is sparse  
  if sp.issparse(A) & (not HAS_CHOLMOD):
    warnings.warn(
      'Could not import CHOLMOD. Sparse matrices will be converted to '
      'dense for all Cholesky decompositions. To install CHOLMOD and its '
      'python wrapper, follow the instructions at '
      'https://scikit-sparse.readthedocs.io')
    A = A.toarray()    

  if sp.issparse(A):
    # Cholesky decomposition for sparse matrix using CHOLMOD
    A = A.tocsc()
    try:
      factor = cholmod.cholesky(A)
      L = factor.L()
      perm = factor.P()

    except cholmod.CholmodNotPositiveDefiniteError as err:  
      raise NotPositiveDefiniteError(err.args[0])
  
  else:
    # Cholesky decomposition for numpy array using LAPACK
    L = _lapack_cholesky(A,lower=True)
    # make permutation matrix an idenity matrix
    perm = np.arange(n).astype(np.int32)

  return L,perm


def _is_positive_definite(A):
  ''' 
  Tests if *A* is positive definite. This is done by testing whether
  the Cholesky decomposition finishes successfully
  '''
  try:
    _cholesky(A)

  except NotPositiveDefiniteError:  
    return False
  
  return True


class _InversePermutedTriangular(object):
  ''' 
  Emulates the inverse of a permuted lower or upper triangular matrix.
  Lower triangular matrices are permuted as *P.T.dot(L)*, and upper
  triangular matrices are permuted as *U.dot(P.T)*. The user specifies
  *L* and the permutation indices *perm*. 
  
  Note
  ----
  These two code blocks should produce equivalent results

    >>> L,perm = _cholesky(A)
    >>> PTLinv = _InversePermutedTriangular(L,perm,lower=True)
    >>> PTLinv.dot(np.random.random(A.shape[0]))
  
    >>> L,perm = _cholesky(A)
    >>> perm_inv = np.argsort(perm)
    >>> PTLinv = np.linalg.inv(L[perm_inv,:])
    >>> PTLinv.dot(np.random.random(A.shape[0]))

  ''' 
  def __init__(self,L,perm,lower=True,build_inverse=False):
    ''' 
    Parameters
    ----------
    L : (N,N) array or (N,N) sparse matrix

    perm : (N,) int array
      permutations. A[perm] == P.dot(A)
      
    lower : bool
    
    build_inverse : bool, optional
      actually construct the inverse
    '''   
    if sp.issparse(L):
      # csr is more efficient for the sparse solver
      L = L.tocsr()
        
    self.L = L
    self.perm = perm
    self.lower = lower
    if build_inverse:
      self._inverse = self.dot(np.eye(L.shape[0]))

  def dot(self,b):
    ''' 
    Parameters
    ----------
    b : (N,*) array
    '''
    # if *b* is a sparse matrix convert it to an array. Otherwise the
    # solvers may not understand the input.
    if sp.issparse(b):
      b = b.toarray()
    
    if hasattr(self,'_inverse'):
      # if the inverse has been built then use it
      out = self._inverse.dot(b)
      
    else:
      # permutation is done on the right side if lower 
      if self.lower:
        b = b[self.perm]
       
      if sp.issparse(self.L):
        #out = spla.spsolve(self.L,b)
        out = spla.spsolve_triangular(self.L,b,lower=self.lower)
      
      else:
        out = _lapack_solve_triangular(self.L,b,lower=self.lower)    
      
      # permutation is done on the left side if upper
      if not self.lower:
        out = out[self.perm]
        
    return out


class _InversePositiveDefinite(object):
  ''' 
  Emulates the inverse of a positive definite matrix *A*, which is
  internally decomposed as

      A^-1 = (L^T * P)^-1 * (P^T * L)^-1
             ------------   ------------
              upper tri.     lower tri.
  '''
  def __init__(self,A,build_inverse=False):
    ''' 
    Parameters
    ----------
    A : (N,N) numpy array or sparse matrix

    build_inverse : bool, optional
      actually construct the inverse
    '''
    L,perm = _cholesky(A)
    perm_inv = np.argsort(perm) 
    self.lower_inv = _InversePermutedTriangular(L,perm,lower=True)
    self.upper_inv = _InversePermutedTriangular(L.T,perm_inv,lower=False)
    if build_inverse:
      self._inverse = self.dot(np.eye(A.shape[0]))
  
  def dot(self,b):
    if sp.issparse(b):
      b = b.toarray()

    if hasattr(self,'_inverse'):
      # if the inverse has been built then use it
      out = self._inverse.dot(b)

    else:
      out = self.upper_inv.dot(self.lower_inv.dot(b))

    return out
    

class _InversePartitioned(object):
  ''' 
  Emulates the inverse of the partitioned matrix
  
     |  A   B  |
     | B.T  0  |,

  where A is a positive definite matrix. The partitioned inverse can
  be written as
  
     |  C   D  |
     | D.T  E  |
     
  where 
  
    C = A^-1 - (A^-1 * B) * (B^T * A^-1 * B)^-1 * (A^-1 * B)^T
    
    D = (A^-1 * B) * (B^T * A^-1 * B)^-1
    
    E = - (B^T * A^-1 * B)^-1
  
  '''
  def __init__(self,A,B,build_inverse=False):
    ''' 
    Parameters
    ----------
    A : (N,N) array or sparse matrix
    B : (N,P) array 
    '''
    # B should never be sparse... but just incase
    if sp.issparse(B):
      B = B.toarray()

    n,p = B.shape
    if n < p:
      raise np.linalg.LinAlgError(
        'There are fewer rows than columns in *B*. This makes the '
        'block matrix singular, and its inverse cannot be computed.')
    
    Ainv = _InversePositiveDefinite(A,build_inverse=build_inverse)  
    AinvB = Ainv.dot(B) 
    E = -_lapack_cholesky_inv(B.T.dot(AinvB)) 
    D = -AinvB.dot(E) 
    # NOTE that AinvB, E, and D are all dense arrays since they
    # relatively smaller than C
    self.Ainv = Ainv
    self.AinvB = AinvB
    self.E = E
    self.D = D
    
  def dot(self,a,b):   
    ''' 
    Dot an (N,*) array and (P,*) array with the inverse matrix. The
    solution is returned in two parts.
    '''
    if sp.issparse(a):
      a = a.toarray()

    if sp.issparse(b):
      b = b.toarray()

    out1  = (self.Ainv.dot(a) - 
             self.D.dot(self.AinvB.T.dot(a)) +
             self.D.dot(b))
    out2  = self.D.T.dot(a) + self.E.dot(b)

        
    return out1,out2
    

def _sample(mean,cov,use_cholesky=False):
  ''' 
  Draws a random sample from the Gaussian process with the specified 
  mean and covariance. 
  '''   
  if use_cholesky:
    # draw a sample using a cholesky decomposition. This assumes that
    # *cov* is numerically positive definite (i.e. no small negative
    # eigenvalues from rounding error).
    L,perm = _cholesky(cov)
    # perm_inv is effectively P.T
    perm_inv = np.argsort(perm)
    w = np.random.normal(0.0,1.0,mean.shape[0])
    u = mean + L[perm_inv].dot(w)
  
  else:
    # otherwise use an eigenvalue decomposition, ignoring negative
    # eigenvalues. If *cov* is sparse then begrudgingly make it dense.
    # TODO: an LDL^T decomposition would be better suited for this.
    if sp.issparse(cov):
      cov = cov.toarray()
      
    s,Q = np.linalg.eigh(cov)
    keep = (s > 0.0)
    w = np.random.normal(0.0,np.sqrt(s[keep]))
    u = mean + Q[:,keep].dot(w)
     
  return u


class Memoize(object):
  ''' 
  Memoizing decorator. The output for calls to decorated functions
  will be cached and reused if the function is called again with the
  same arguments. This is intendend to decorate the mean, covariance,
  and basis functions for *GaussianProcess* instances.

  Parameters
  ----------
  fin : function
    Function that takes numpy arrays as input.
  
  Returns
  -------
  fout : function
    Memoized function.
  
  Notes
  -----
  1. Caches can be cleared with the module-level function
  *clear_caches*.
      
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
      # make sure there is room for the new entry
      while len(self.cache) >= Memoize.MAX_CACHE_SIZE:
        self.cache.popitem(0)
        
      self.cache[key] = self.fin(*args)
      
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


def _condition(gp,y,d,sigma,p,obs_diff):
  '''   
  Returns a conditioned *GaussianProcess*.
  '''
  @Memoize
  def precompute():
    # do as many calculations as possible without yet knowning where
    # the interpolation points will be. This function is memoized so
    # that I can easily dereference the kernel inverse matrix with
    # "clear_caches".
    logger.debug('Calculating and caching kernel inverse ...')
    # GP mean at the observation points
    mu_y = gp._mean(y,obs_diff)
    # GP covariance at the observation points
    C_y = gp._covariance(y,y,obs_diff,obs_diff)
    # GP basis functions at the observation points
    p_y = gp._basis(y,obs_diff)    
    # add data noise to the covariance matrix
    C_y = C_y + sigma
    # append the data noise basis vectors 
    p_y = np.hstack((p_y,p)) 

    if sp.issparse(C_y):
      logger.debug('Kernel is sparse with %2.3f%% non-zeros' % 
                   (C_y.nnz/(1.0*np.prod(C_y.shape))))

    K_y_inv = _InversePartitioned(C_y,p_y)
    r  = d - mu_y
    logger.debug('Done')
    return K_y_inv,r
    
  def mean(x,diff):
    K_y_inv,r = precompute()
    mu_x = gp._mean(x,diff)
    C_xy = gp._covariance(x,y,diff,obs_diff)

    # pad p_x with as many zero columns as there are noise basis vectors
    p_x = gp._basis(x,diff)
    p_x_pad = np.zeros((p_x.shape[0],p.shape[1]),dtype=float)
    p_x = np.hstack((p_x,p_x_pad))

    vec1,vec2 = K_y_inv.dot(r,np.zeros(p_x.shape[1]))
    out = mu_x + C_xy.dot(vec1) + p_x.dot(vec2)
    return out

  def covariance(x1,x2,diff1,diff2):
    K_y_inv,r = precompute()
    C_x1x2 = gp._covariance(x1,x2,diff1,diff2)
    C_x1y = gp._covariance(x1,y,diff1,obs_diff)
    C_x2y = gp._covariance(x2,y,diff2,obs_diff)

    p_x1 = gp._basis(x1,diff1)
    p_x1_pad = np.zeros((p_x1.shape[0],p.shape[1]),dtype=float)
    p_x1 = np.hstack((p_x1,p_x1_pad))

    p_x2 = gp._basis(x2,diff2)
    p_x2_pad = np.zeros((p_x2.shape[0],p.shape[1]),dtype=float)
    p_x2 = np.hstack((p_x2,p_x2_pad))

    mat1,mat2 = K_y_inv.dot(C_x2y.T,p_x2.T)
    out = C_x1x2 - C_x1y.dot(mat1) - p_x1.dot(mat2)
    return out
  
  dim = y.shape[1]
  out = GaussianProcess(mean,covariance,dim=dim)
  return out



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
  
  sigma : (N,) array, (N,N) array, or (N,N) scipy sparse matrix    
    If this is an (N,) array then it describes one standard deviation
    of the random vector. If this is an (N,N) array then it describes
    the covariances.
  
  p : (N,P) array, optional 
    Improper basis vectors. If specified, then *d* is assumed to
    contain some unknown linear combination of the columns of *p*.

  Notes
  -----
  Unlike other functions in this module, if the covariance matrix is
  not numerically positive definite then this function will fail with
  an error rather than trying to coerce it into a positive definite
  matrix.

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
  
  sigma = _asarray(sigma,dtype=float)
  if sigma.ndim == 1:
    # convert std. dev. to a csc sparse covariance matrix
    sigma = sp.diags(sigma**2).tocsc()
    
  _assert_shape(sigma,(d.shape[0],d.shape[0]),'sigma')

  if p is None:
    p = np.zeros((d.shape[0],0),dtype=float)
  
  else:  
    p = np.asarray(p,dtype=float)
  
  _assert_shape(p,(d.shape[0],None),'p')
  
  n,m = p.shape
  A,perm = _cholesky(sigma)
  PTAinv = _InversePermutedTriangular(A,perm)

  B = PTAinv.dot(p)

  C,perm = _cholesky(B.T.dot(B))   
  PTCinv = _InversePermutedTriangular(C,perm)

  D,perm = _cholesky(p.T.dot(p))   

  a = PTAinv.dot(d - mu)
  b = PTCinv.dot(B.T.dot(a))

  out = (np.sum( np.log( D.diagonal() ) ) -
         np.sum( np.log( A.diagonal() ) ) -
         np.sum( np.log( C.diagonal() ) ) -
         0.5*a.T.dot(a) +
         0.5*b.T.dot(b) -
         0.5*(n-m)*np.log(2*np.pi))
  return out


def outliers(d,s,mu=None,sigma=None,p=None,tol=4.0):
  ''' 
  Uses a data editing algorithm to identify outliers in *d*. Outliers
  are considered to be the data that are abnormally inconsistent with
  the Gaussian process described by *mu* (mean), *sigma* (covariance),
  and *p* (basis vectors). This function can only be used for data
  with nonzero, uncorrelated noise.

  The data editing algorithm first conditions the Gaussian process
  with the observations, then it compares each residual (*d* minus the
  expected value of the posterior divided by *sigma*) to the RMS of
  residuals. Data with residuals greater than *tol* times the RMS are
  identified as outliers. This process is then repeated using the
  subset of *d* which were not flagged as outliers. If no new outliers
  are detected in an iteration then the algorithm stops.

  Parameters
  ----------  
  d : (N,) float array
    Observations
  
  s : (N,) float array
    One standard deviation uncertainty on the observations. 
  
  mu : (N,) float array, optional
    Mean of the Gaussian process at the observation points. Defaults
    to zeros.

  sigma : (N,N) float array or (N,N) scipy sparse matrix, optional
    Covariance of the Gaussian process at the observation points.
    Defaults to zeros.
  
  p : (N,P) float array, optional
    Improper basis vectors for the Gaussian process evaluated at the
    observation points. Defaults to an (N,0) array.
  
  tol : float, optional
    Outlier tolerance. Smaller values make the algorithm more likely
    to identify outliers. A good value is 4.0 and this should not be
    set any lower than 2.0.
  
  Returns
  -------
  out : (N,) bool array
    Array indicating which data are outliers

  '''
  d = np.asarray(d,dtype=float)
  _assert_shape(d,(None,),'d')
  n = d.shape[0]

  s = np.asarray(s,dtype=float)
  _assert_shape(s,(n,),'s')

  if mu is None:
    mu = np.zeros((n,),dtype=float)
  
  else:  
    mu = np.asarray(mu,dtype=float)
    _assert_shape(mu,(n,),'mu')

  if sigma is None:
    sigma = sp.csc_matrix((n,n),dtype=float)
  
  else:
    sigma = _asarray(sigma,dtype=float)
    _assert_shape(sigma,(n,n),'sigma')
  
  if p is None:  
    p = np.zeros((n,0),dtype=float)
  
  else:  
    p = np.asarray(p,dtype=float)
    _assert_shape(p,(n,None),'p')
  
  # number of basis functions
  m = p.shape[1]
  # total number of outlier detection iterations
  itr = 1
  # boolean array indicating outliers
  out = np.zeros(n,dtype=bool)
  while True:
    logger.debug('Starting iteration %s of outlier detection routine' % itr)
    # remove rows and cols where *out* is True
    sigma_i = sigma[:,~out][~out,:]
    p_i = p[~out]
    mu_i = mu[~out]
    d_i = d[~out]
    s_i = s[~out]
    # add data covariance to GP covariance
    sigma_i = sigma_i + sp.diags(s_i**2).tocsc()
    Kinv = _InversePartitioned(sigma_i,p_i)
    vec1,vec2 = Kinv.dot(d_i - mu_i,np.zeros(m))

    # dereference everything that we no longer need
    del sigma_i,mu_i,p_i,d_i,s_i,Kinv
    
    fit = mu + sigma[:,~out].dot(vec1) + p.dot(vec2)
    # find new outliers
    res = np.abs(fit - d)/s
    rms = np.sqrt(np.mean(res[~out]**2))
    if np.all(out == (res > tol*rms)):
      break

    else:
      out = res > tol*rms
      itr += 1

  logger.debug('Detected %s outliers out of %s observations' %
               (sum(out),len(out)))

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
  return sp.csc_matrix((x1.shape[0],x2.shape[0]),dtype=float)  


def _empty_basis(x,diff):
  '''empty set of basis functions'''
  return np.zeros((x.shape[0],0),dtype=float)  
  

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
    return an (N,) numpy array.

  covariance : function
    Covariance function for the Gaussian process. This takes either
    two arguments, *x1* and *x2*, or four arguments, *x1*, *x2*,
    *diff1* and *diff2*. *x1* and *x2* are (N,D) and (M,D) arrays of
    positions, respectively. *diff1* and *diff2* are (D,) arrays
    specifying the derivatives with respect to *x1* and *x2*,
    respectively. If the function only takes two arguments, then the
    function is assumed to not be differentiable. The function should
    return an (N,M) numpy array or an (N,M) scipy sparse matrix.

  basis : function, optional
    Improper basis functions. This function takes either one argument,
    *x*, or two arguments, *x* and *diff*. *x* is an (N,D) array of
    positions and *diff* is a (D,) array specifying the derivative.
    This function should return an (N,P) numpy array, where each
    column is a basis function evaluated at *x*. By default, a
    *GaussianProcess* instance contains no improper basis functions.
        
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
      
    sigma : (N,) array, (N,N) array, or (N,N) scipy sparse matrix, optional
      Data uncertainty. If this is an (N,) array then it describes one 
      standard deviation of the data error. If this is an (N,N) array 
      then it describes the covariances of the data error. If nothing 
      is provided then the error is assumed to be zero. Note that 
      having zero uncertainty can result in numerically unstable 
      calculations for large N.

    p : (N,P) array, optional  
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
      sigma = sp.csc_matrix((n,n),dtype=float)
    
    else:
      sigma = _asarray(sigma,dtype=float)
      if sigma.ndim == 1:
        # convert standard deviations to covariances
        sigma = sp.diags(sigma**2).tocsc()

      _assert_shape(sigma,(n,n),'sigma')
        
    if p is None:
      p = np.zeros((n,0),dtype=float)
    
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

  def likelihood(self,y,d,sigma=None,p=None):
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
      sigma = sp.csc_matrix((n,n),dtype=float)

    else:
      sigma = _asarray(sigma,dtype=float)
      if sigma.ndim == 1:
        # If sigma is a 1d array then it contains std. dev. uncertainties.  
        # Convert sigma to a covariance matrix
        sigma = sp.diags(sigma**2).tocsc()

      _assert_shape(sigma,(n,n),'sigma')
    
    if p is None:
      p = np.zeros((n,0),dtype=float)

    else:
      p = np.asarray(p,dtype=float)
      _assert_shape(p,(n,None),'p')

    obs_diff = np.zeros(dim,dtype=int)

	  # find the mean, covariance, and improper basis for the combination
  	# of the Gaussian process and the noise.
    mu = self._mean(y,obs_diff)
    sigma = self._covariance(y,y,obs_diff,obs_diff) + sigma
    p = np.hstack((self._basis(y,obs_diff),p))
    out = likelihood(d,mu,sigma,p=p)
    return out

  def outliers(self,y,d,sigma,tol=4.0):  
    ''' 
    Uses a data editing algorithm to identify outliers in *d*.
    Outliers are considered to be the data that are abnormally
    inconsistent with the *GaussianProcess*. This method can only be
    used for data that has nonzero, uncorrelated noise.

    The data editing algorithm first conditions the *GaussianProcess*
    with the observations, then it compares each residual (*d* minus
    the expected value of the posterior divided by *sigma*) to the RMS
    of residuals. Data with residuals greater than *tol* times the RMS
    are identified as outliers. This process is then repeated using
    the subset of *d* which were not flagged as outliers. If no new
    outliers are detected in an iteration then the algorithms stops.
    
    Parameters
    ----------
    y : (N,D) float array
      Observation points.
    
    d : (N,) float array
      Observed values at *y*
    
    sigma : (N,) float array
      One standard deviation uncertainty on *d* 
    
    tol : float
      Outlier tolerance. Smaller values make the algorithm more likely
      to identify outliers. A good value is 4.0 and this should not be
      set any lower than 2.0.
    
    Returns
    -------
    out : (N,) bool array
      Boolean array indicating which data are outliers
    
    '''
    y = np.asarray(y,dtype=float)
    _assert_shape(y,(None,self.dim),'y')
    n,dim = y.shape # number of observations and dimensions

    d = np.asarray(d,dtype=float)
    _assert_shape(d,(n,),'d')

    sigma = np.asarray(sigma,dtype=float)
    _assert_shape(sigma,(n,),'sigma')
    
    obs_diff = np.zeros(dim,dtype=int)
   
	  # find the mean, covariance, and improper basis for the combination
  	# of the Gaussian process and the noise.
    gp_mu = self._mean(y,obs_diff)
    gp_sigma = self._covariance(y,y,obs_diff,obs_diff)
    gp_p = self._basis(y,obs_diff)
    out = outliers(d,sigma,
                   mu=gp_mu,sigma=gp_sigma,
                   p=gp_p,tol=tol)
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
    # return a dense copy of out
    if sp.issparse(out):
      out = out.toarray()

    else:
      out = np.array(out,copy=True)
      
    return out

  def mean(self,x,diff=None):
    ''' 
    Returns the mean of the proper component of the *GaussianProcess*.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
        
    diff : (D,) tuple
      Derivative specification    
      
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
    # return a dense copy of out
    if sp.issparse(out):
      out = out.toarray()

    else:
      out = np.array(out,copy=True)

    return out

  def covariance(self,x1,x2,diff1=None,diff2=None):
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
    # return a dense copy of out
    if sp.issparse(out):
      out = out.toarray()

    else:
      out = np.array(out,copy=True)

    return out
    
  def meansd(self,x,chunk_size=100):
    ''' 
    Returns the mean and standard deviation of the proper component of
    the *GaussianProcess*. This does not return the full covariance
    matrix, making it appropriate for evaluating the *GaussianProcess*
    at many points.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    chunk_size : int, optional  
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
    x = np.asarray(x,dtype=float)
    _assert_shape(x,(None,self.dim),'x')
    # derivative of output will be zero
    diff = np.zeros(x.shape[1],dtype=int)

    count = 0
    xlen = x.shape[0]
    out_mean = np.zeros(xlen,dtype=float)
    out_sd = np.zeros(xlen,dtype=float)
    while count < xlen:
      # only log the progress if the mean and sd are being build in
      # multiple chunks
      if xlen > chunk_size:
        logger.debug('Computing the mean and std. dev. : %2.1f%% complete' % ((100.0*count)/xlen))
      
      start,stop = count,count+chunk_size 
      out_mean[start:stop] = self._mean(x[start:stop],diff) 
      cov = self._covariance(x[start:stop],x[start:stop],diff,diff) 
      var = cov.diagonal()
      out_sd[start:stop] = np.sqrt(var) 
      count += chunk_size
    
    if xlen > chunk_size:
      logger.debug('Computing the mean and std. dev. : 100% complete')

    return out_mean,out_sd

  def sample(self,x,c=None,use_cholesky=False):  
    '''  
    Draws a random sample from the *GaussianProcess*.  
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points.
    
    c : (P,) array, optional
      Coefficients for the improper basis functions. If this is not
      specified then they are set to zero.
    
    use_cholesky : bool, optional
      Indicates whether to use the Cholesky decomposition to create
      the sample. The Cholesky decomposition is faster but it assumes
      that the covariance matrix is numerically positive definite
      (i.e. there are no slightly negative eigenvalues due to rounding
      error).
      
    Returns
    -------
    out : (N,) array      
    
    '''
    x = np.asarray(x,dtype=float)
    _assert_shape(x,(None,self.dim),'x')
    # derivative of the sample will be zero
    diff = np.zeros(x.shape[1],dtype=int)

    mu = self._mean(x,diff)
    sigma = self._covariance(x,x,diff,diff)
    p = self._basis(x,diff)
    
    if c is not None:
      c = np.asarray(c,dtype=float)
    
    else:
      c = np.zeros(p.shape[1])  

    _assert_shape(c,(p.shape[1],),'c')    

    out = _sample(mu,sigma,use_cholesky=use_cholesky) + p.dot(c)
    return out
    
  def is_positive_definite(self,x):
    '''     
    Tests if the covariance matrix, which is the covariance function
    evaluated at *x*, is positive definite. This is done by testing if
    the Cholesky decomposition of the covariance matrix finishes
    successfully. 
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points

    Returns
    -------
    out : bool


    Notes
    -----
    1. This function may return *False* even if the covariance
    function is positive definite. This is because some of the
    eigenvalues for the matrix are so small that they become slightly
    negative due to numerical rounding error. This is most notably the
    case for the squared exponential covariance function.    
    '''
    x = np.asarray(x,dtype=float)
    _assert_shape(x,(None,self.dim),'x')
    diff = np.zeros(x.shape[1],dtype=int)

    cov = self._covariance(x,x,diff,diff)    
    out = _is_positive_definite(cov)
    return out  
    
  def memoize(self):
    ''' 
    Memoizes the *_mean*, *_covariance*, and *_basis* methods for this
    *GaussianProcess*. This can improve performance by cutting out
    redundant computations, but it may also increase memory
    consumption.
    '''
    self._mean = Memoize(self._mean)
    self._covariance = Memoize(self._covariance)
    self._basis = Memoize(self._basis)


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
      
  def covariance(x1,x2,diff1,diff2):
    a,b,c = params  
    diff = diff1 + diff2
    out = b*(-1)**sum(diff2)*phi(x1,x2,eps=c,diff=diff)
    if not _all_is_finite(out):
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
  
  Notes
  -----
  1. Some of the eigenvalues for squared exponential covariance
  matrices are very small and may be slightly negative due to
  numerical rounding error. Consequently, the Cholesky decomposition
  for a squared exponential covariance matrix will often fail. This
  becomes a problem when conditioning a squared exponential
  *GaussianProcess* with noise-free data.
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
  
  sigma : (P,) or (P,P) array w
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
  def basis(x,diff):
    powers = rbf.poly.powers(order,x.shape[1])
    out = rbf.poly.mvmonos(x,powers,diff)
    return out
  
  out = gpbfci(basis,dim=dim)  
  return out
