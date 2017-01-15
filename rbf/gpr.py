''' 
This module defines a class, *GaussianProcess*, which is an 
abstraction that allows one to easily work with Gaussian processes. 
The *GaussianProcess* class is primarily intended for Gaussian process 
regression (GPR), which is performed with the *condition* method. GPR 
is a technique for constructing a continuous function from discrete 
and potentially noisy observations. This documentation describes 
Gaussian processes and the operations (methods), which they are 
endowed with. Details on the classes *GaussianProcess* and 
*PriorGaussianProcess* can be found in their doc strings.

There are several existing python packages for GPR. Some packages are 
well developed and contain a great deal of functionilty which is 
absent here. For example, this module does not contain any routines 
for optimizing hyperparameters. However, this module is not a stripped 
down rewrite of existing packages. Instead, this module approaches GPR 
from an object oriented perspective. GPR is treated as a method of a 
*GaussianProcess* and the method returns a new *GaussianProcess* which 
can itself be used as a prior for further GPR. *GaussianProcess* 
instances also have methods for addition, subtraction, scaling, and 
differentiation, which each return a *GaussianProcess* possessing the 
same methods. This object oriented approach is intended to give the 
user the flexibility necessary for data analysis with Gaussian 
processes.

Gaussian Processes
==================
A Gaussian process is a stochastic process, :math:`u(x)`, which has a 
domain in :math:`\mathbb{R}^n` and we define it in terms of a mean 
function, :math:`\\bar{u}(x)`, a covariance function,
:math:`C_u(x,x')`, and the order of the polynomial null space, 
:math:`d_u`. The null space is the span of all monomial
basis functions in :math:`\mathbb{R}^n` which have order up to and 
including :math:`d_u`. These monomials are denoted as
:math:`\mathbf{p}_u(x) = [p_i(x)]_{i=1}^{m_u}`, where :math:`m_u = 
{{n+d_u}\choose{n}}`. It is not necessary for a Gaussian process to 
have a null space. If there is no null space then we say 
:math:`d_u=-1`. We express the Gaussian process as
  
.. math::
  u(x) = u_o(x) + \sum_{i=1}^{m_u} c_i p_i(x),

where :math:`\{c_i\}_{i=1}^{m_u}` are uncorrelated random variables 
with infinite variance and

.. math::
  u_o \\sim \\mathcal{N}\\left(\\bar{u},C_u\\right).

We consider five operations on Gaussian processes: addition, 
subtraction, scaling, differentiation, and conditioning. Each 
operation produces another Gaussian process which possesses the same 
five operations. These operations are described below.

Operations on Gaussian Processes
================================

Addition
--------
Two uncorrelated Gaussian processes, :math:`u` and :math:`v`, can be 
added as

.. math::
  u(x) + v(x) = z(x)

where the mean, covariance, and null space order for :math:`z` are

.. math::
  \\bar{z}(x) = \\bar{u}(x) + \\bar{v}(x),

.. math::
  C_z(x,x') = C_u(x,x') + C_v(x,x'),
  
and 

.. math::
  d_z = \max(d_u,d_v).

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
  d_z = \max(d_u,d_v).


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
  d_z = 
  \\begin{cases}
  d_u, &\\text{if  } c\\neq 0   \\\\  
  -1, &\mathrm{otherwise}  \\\\
  \\end{cases}.

Differentiation
---------------
A Gaussian process can be differentiated with the differential 
operator

.. math::
  D_x = \\frac{\partial^{a_1 + a_2 + \dots + a_n}}
              {\partial x_1^{a_1} \partial x_2^{a_2} \dots 
              \partial x_n^{a_n}},

where :math:`\{x_i\}_{i=1}^n` are the basis vectors of 
:math:`\mathbb{R}^n`, as

.. math::
  D_xu(x) = z(x) 

where 

.. math::
  \\bar{z}(x) = D_x\\bar{u}(x),
  
.. math::
  C_z(x,x') = D_xD_{x'}C_u(x,x'),
  
.. math::
  d_z = \max(d_u - d_D,-1),

and :math:`d_D = a_1 + a_2 + \dots + a_n`. 

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
  d_z = -1.

In the above equations we use the augmented covariance matrices, 
:math:`\mathbf{k}` and :math:`\mathbf{K}`, which are defined as

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
suitably padded with zeros. Note that there is no null space in
:math:`z` because it is assumed that there is enough data in 
:math:`\mathbf{d}` to constrain the null spaces in :math:`u`. If 
:math:`\mathbf{d}` is not sufficiently informative then 
:math:`\mathbf{K}(\mathbf{y})` will not be invertible. A necessary but 
not sufficient condition for :math:`\mathbf{K}(\mathbf{y})` to be 
invertible is that :math:`q \geq m_u`.

Prior Gaussian Processes
========================

This module is primarily intended for Gaussian process regression 
(GPR) and we begin a GPR problem by assuming a prior stochastic model 
for the underlying signal which we are trying to uncover. In this 
module, priors are stationary Gaussian processes which have mean and 
covariance functions described as
  
.. math::
  \\bar{u}(x) = a,
  
and

.. math::
  C_u(x,x') = b\phi\\left(\\frac{||x - x'||_2}{c}\\right), 
  
where :math:`a`, :math:`b`, and :math:`c` are user specified 
coefficients. The literature on radial basis functions and Gaussian 
process regression often refers to :math:`c` as the shape parameter or 
the characteristic length scale. :math:`\phi` is a user specified 
positive definite radial function. One common choice for :math:`\phi` 
is the squared exponential function,

.. math::
  \phi(r) = \exp(-r^2),

which has the benefit of being infinitely differentiable. See [1] for 
a list of commonly used radial functions as well as for more 
information on Gaussian processes.

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
import sympy
import logging
logger = logging.getLogger(__name__)

  
def _is_positive_definite(A,tol=1e-10):
  ''' 
  Tests if *A* is a positive definite matrix. This function returns 
  True if *A* is symmetric and all of its eigenvalues are real and 
  positive.  
  '''
  # test if A is symmetric
  if np.any(np.abs(A - A.T) > tol):
    return False
    
  val,_ = np.linalg.eig(A)
  # test if all the eigenvalues are real 
  if np.any(np.abs(val.imag) > tol):
    return False
    
  # test if all the eigenvalues are positive
  if np.any(val.real < -tol):
    return False

  return True  


def _draw_sample(mean,cov,tol=1e-10):
  ''' 
  Draws a random sample from the gaussian process with the specified 
  mean and covariance.
  '''
  mean = np.asarray(mean)
  cov = np.asarray(cov)
  val,vec = np.linalg.eigh(cov)
  # ignore any slightly imaginary components
  val = val.real
  vec = vec.real
  # indices of positive eigenvalues
  idx = val > 0.0
  # generate independent normal random numbers with variance equal to 
  # the eigenvalues
  sample = np.random.normal(0.0,np.sqrt(val[idx]))
  # map with the eigenvectors and add the mean
  sample = mean + vec[:,idx].dot(sample)
  return sample


def _sigfigs(val,n):
  ''' 
  Returns *val* rounded to *n* significant figures. This is just for 
  display purposes.
  '''
  if val == 0.0:
    return 0.0
  
  if ~np.isfinite(val):
    return val
    
  d = -np.int(np.log10(np.abs(val))) + n - 1 
  out = np.round(val,d)
  return out
    

class _Memoize(object):
  ''' 
  Memoizing decorator. This works for functions that take only 
  positional arguments and each argument is a numpy array.
  '''
  # static variable controlling the maximum cache size for all 
  # memoized functions
  MAX_CACHE_SIZE = 500
  
  def __init__(self,fin):
    self.cache = OrderedDict()
    self.fin = fin
    self.links = []

  def __call__(self,*args):
    ''' 
    Calls *fin* if *fin(*args)* is not stored in the cache. If 
    *fin(*args)* is in the cache then return it without evaluating 
    *fin*.
    '''
    # it is assumed that all the arguments are numpy arrays
    key = tuple(a.tobytes() for a in args)
    if key not in self.cache:
      output = self.fin(*args)
      # make output read-only. This prevents the end-user from 
      # inadvertently modifying the entries in the cache.
      #output.flags['WRITEABLE'] = False
      # make sure there is room for the new entry
      while len(self.cache) >= _Memoize.MAX_CACHE_SIZE:
        self.cache.popitem(0)
        
      self.cache[key] = output
 
    return self.cache[key]

  def __repr__(self):
    return self.fin.__repr__()

  def add_links(self,*args):    
    ''' 
    Link other memoized functions so that calling *clear_cache* for 
    this function also calls *clear_cache* for the linked functions.
    '''
    # append new links to existing links
    links = self.links + list(args)
    # make sure there are no duplicate links
    links = list(set(links))
    self.links = links

  def clear_cache(self):    
    ''' 
    Dereference the current cache and instantiate an empty one. This 
    is also done for any linked functions.
    '''
    self.cache = OrderedDict()
    for l in self.links:
      if hasattr(l,'clear_cache'):
        l.clear_cache()
    
    
def get_max_cache_size():
  ''' 
  Returns the maximum cache size for memoized functions.
  '''
  return _Memoize.MAX_CACHE_SIZE
  

def set_max_cache_size(val):
  ''' 
  Sets the maximum cache size for memoized functions. Increasing this 
  value generally improves speed at the expense of memory efficiency.
  '''
  val = int(val)
  if val < 0:
    raise ValueError('maximum cache size must be positive')
    
  _Memoize.MAX_CACHE_SIZE = val


@_Memoize
def _mvmonos(x,powers,diff):
  ''' 
  Memoized function which returns the matrix of monomials spanning the 
  null space
  '''
  return rbf.poly.mvmonos(x,powers,diff)


def _add_factory(gp1,gp2):
  '''   
  Factory function which returns the mean and covariance functions for 
  two added *GaussianProcesses*.
  '''
  @_Memoize
  def mean(x,diff):
    out = gp1._mean(x,diff) + gp2._mean(x,diff)
    return out       

  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out
            
  # link all functions called by *mean*
  mean.add_links(gp1._mean,gp2._mean) 
  # link all functions called by *covariance*
  covariance.add_links(gp1._covariance,gp2._covariance) 
  return mean,covariance
  

def _subtract_factory(gp1,gp2):
  '''   
  Factory function which returns the mean and covariance functions for 
  a *GaussianProcess* which has been subtracted from another 
  *GaussianProcess*.
  '''
  @_Memoize
  def mean(x,diff):
    out = gp1._mean(x,diff) - gp2._mean(x,diff)
    return out
      
  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out       
            
  mean.add_links(gp1._mean,gp2._mean) 
  covariance.add_links(gp1._covariance,gp2._covariance) 
  return mean,covariance


def _scale_factory(gp,c):
  '''   
  Factory function which returns the mean and covariance functions for 
  a scaled *GaussianProcess*.
  '''
  @_Memoize
  def mean(x,diff):
    out = c*gp._mean(x,diff)
    return out

  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    out = c**2*gp._covariance(x1,x2,diff1,diff2)
    return out
      
  mean.add_links(gp._mean) 
  covariance.add_links(gp._covariance) 
  return mean,covariance


def _differentiate_factory(gp,d):
  '''   
  Factory function which returns the mean and covariance functions for 
  a differentiated *GaussianProcess*.
  '''
  @_Memoize
  def mean(x,diff):
    out = gp._mean(x,diff + d)
    return out 

  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    out = gp._covariance(x1,x2,diff1+d,diff2+d)
    return out
      
  mean.add_links(gp._mean) 
  covariance.add_links(gp._covariance) 
  return mean,covariance


def _condition_factory(gp,y,d,sigma,obs_diff):
  '''   
  Factory function which returns the mean and covariance functions for 
  a conditioned *GaussianProcess*.
  '''
  @_Memoize
  def precompute():
    ''' 
    do as many calculations as possible without yet knowning where the 
    interpolation points will be. This is a memoized function because 
    I can then control when the precomputed data is dereferenced.
    '''
    # compute K_y_inv
    Cd = np.diag(sigma**2)
    Cu_yy = gp._covariance(y,y,obs_diff,obs_diff)
    p_y   = gp._null_space(y,obs_diff)
    q,m = d.shape[0],p_y.shape[1]
    K_y = np.zeros((q+m,q+m))
    K_y[:q,:q] = Cu_yy + Cd
    K_y[:q,q:] = p_y
    K_y[q:,:q] = p_y.T
    try:
      K_y_inv = np.linalg.inv(K_y)
      
    except np.linalg.LinAlgError:
      raise np.linalg.LinAlgError(
        'Failed to compute the inverse of K. This could be because '
        'there is not enough data to constrain a null space. This '
        'error could also be caused by noise-free observations that '
        'are inconsistent with the Gaussian process.')

    # compute r
    r = np.zeros(q+m)
    r[:q] = d - gp._mean(y,obs_diff)
    return K_y_inv,r
    
  @_Memoize
  def mean(x,diff):
    K_y_inv,r = precompute()
    Cu_xy = gp._covariance(x,y,diff,obs_diff)
    p_x   = gp._null_space(x,diff)
    k_xy  = np.hstack((Cu_xy,p_x))
    out   = gp._mean(x,diff) + k_xy.dot(K_y_inv.dot(r))
    return out

  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    K_y_inv,r = precompute()
    Cu_x1x2 = gp._covariance(x1,x2,diff1,diff2)
    Cu_x1y  = gp._covariance(x1,y,diff1,obs_diff)
    Cu_x2y  = gp._covariance(x2,y,diff2,obs_diff)
    p_x1    = gp._null_space(x1,diff1)
    p_x2    = gp._null_space(x2,diff2)
    k_x1y   = np.hstack((Cu_x1y,p_x1))
    k_x2y   = np.hstack((Cu_x2y,p_x2))
    out = Cu_x1x2 - k_x1y.dot(K_y_inv).dot(k_x2y.T) 
    return out

  # link all functions called by *mean*
  mean.add_links(gp._mean,gp._covariance,precompute) 
  # link all functions called by *covariance*
  covariance.add_links(gp._mean,gp._covariance,precompute) 
  return mean,covariance


def _prior_factory(basis,coeff,order):
  ''' 
  Factory function which returns the mean and covariance functions for 
  a *PriorGaussianProcess*.
  '''
  @_Memoize
  def mean(x,diff):
    a,b,c = coeff  
    out = np.zeros(x.shape[0])
    if sum(diff) == 0:
      out[:] = a

    return out
      
  @_Memoize
  def covariance(x1,x2,diff1,diff2):
    a,b,c = coeff  
    diff = diff1 + diff2
    eps = np.ones(x2.shape[0])/c
    out = b*(-1)**sum(diff2)*basis(x1,x2,eps=eps,diff=diff)
    if np.any(~np.isfinite(out)):
      raise ValueError(
        'Encountered a non-finite prior covariance. This may be '
        'because the prior basis function is not sufficiently '
        'differentiable.')

    return out

  return mean,covariance
  

class GaussianProcess(object):
  ''' 
  A *GaussianProcess* instance represents a stochastic process, which 
  is defined in terms of its mean function, covariance function, and 
  polynomial null space. This class allows for basic operations on 
  Gaussian processes which includes addition, subtraction, scaling, 
  differentiation, sampling, and conditioning.
    
  Parameters
  ----------
  mean : function with arguments (*x*, *diff*) 
    Function that returns the *diff* derivative of the mean at *x*. 
    *x* is a (N,D) float array of positions and *diff* is a (D,) 
    integer array indicating the derivative order along each 
    dimension.

  covariance : function with arguments (*x1*, *x2*, *diff1*, *diff2*)
    Function that returns the covariance between the *diff1* 
    derivative at *x1* and the *diff2* derivative at *x2*.

  order : int, optional
    Order of the polynomial null space. If this is -1 then the 
    Gaussian process contains no null space. This should be used if 
    the data contains trends that are well described by a polynomial.
    
  dim : int, optional  
    Specifies the spatial dimensions of the Gaussian process. An 
    error will be raised if the arguments to the *mean* or 
    *covariance* methods conflict with *dim*.
    
  Notes
  -----
  This class does not check whether the specified covariance function 
  is positive definite, making it easy construct an invalid 
  *GaussianProcess* instance. For this reason, this class should not 
  be directly instantiated by the user. Instead, create a 
  *GaussianProcess* with the subclass *PriorGaussianProcess*.
  
  A *GaussianProcess* returned by *add*, *subtract*, *scale*, 
  *differentiate*, and *condition* has a *mean* and *covariance* 
  function which calls the *mean* and *covariance* functions of its 
  parents. For example, if *gp1* and *gp2* are *GaussianProcess* 
  instances then *gp_sum = gp1 + gp2* is another *GaussianProcess* 
  whose *mean* and *covariance* functions make calls to the *mean* and 
  *covariance* functions from its parents, *gp1* and *gp2*. Due to 
  this recursive implementation, the number of generations of children 
  (for lack of a better term) is limited by the maximum recursion 
  depth.

  '''
  def __init__(self,mean,covariance,order=-1,dim=None):
    # the mean and covariance functions are hidden and the mean and 
    # covariance methods should be preferred because the methods 
    # performs necessary argument checks
    self._mean = mean
    self._covariance = covariance
    self.order = order
    self.dim = dim
  
  def _null_space(self,x,diff):
    ''' 
    Returns an array of the monomial basis functions spanning the 
    polynomial null space
    '''
    powers = rbf.poly.powers(self.order,x.shape[1]) 
    out = _mvmonos(x,powers,diff)
    return out 
    
  def __call__(self,*args,**kwargs):
    ''' 
    equivalent to calling *mean_and_uncertainty*
    '''
    return self.mean_and_uncertainty(*args,**kwargs)

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

  def __repr__(self):
    out = ('<GaussianProcess : mean = %s, cov = %s, order = %s>' 
           % (str(self._mean),str(self._covariance),self.order))
    return out


  def add(self,other):
    ''' 
    Adds two Gaussian processes
    
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
          'The number of spatial dimensions for the Gaussian '
          'processes are inconsistent')
        
    mean,covariance = _add_factory(self,other)
    order = max(self.order,other.order)
    out = GaussianProcess(mean,covariance,order=order)
    return out

  def subtract(self,other):
    '''  
    Subtracts two Gaussian processes
    
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
          'The number of spatial dimensions for the Gaussian '
          'processes are inconsistent')

    mean,covariance = _subtract_factory(self,other)
    order = max(self.order,other.order)
    out = GaussianProcess(mean,covariance,order=order)
    return out
    
  def scale(self,c):
    ''' 
    Scales a Gaussian process 
    
    Parameters
    ----------
    c : float
      
    Returns
    -------
    out : GaussianProcess 
      
    '''
    c = np.float64(c)
    mean,covariance = _scale_factory(self,c)
    if c != 0.0:
      order = self.order
    else:
      order = -1
        
    out = GaussianProcess(mean,covariance,order=order)
    return out

  def differentiate(self,d):
    ''' 
    Returns the derivative of a Gaussian process
    
    Parameters
    ----------
    d : (D,) tuple
      Derivative specification
      
    Returns
    -------
    out : GaussianProcess       

    '''
    d = np.asarray(d,dtype=int)
    dim = d.shape[0]
    # if the GaussianProcess already has dim specified then make sure 
    # the derivative specification is consistent
    if self.dim is not None:
      if self.dim != dim:
        raise ValueError(
          'The number of spatial dimensions for *d* is inconsistent '
          'with the Gaussian process.')
          
    mean,covariance = _differentiate_factory(self,d)
    order = max(self.order - sum(d),-1)
    out = GaussianProcess(mean,covariance,dim=dim,order=order)
    return out  

  def condition(self,y,d,sigma=None,obs_diff=None):
    ''' 
    Returns a conditional Gaussian process which incorporates the 
    observed data.
    
    Parameters
    ----------
    y : (N,D) array
      Observation points
    
    d : (N,) array
      Observed values at *y*
      
    sigma : (N,) array, optional
      One standard deviation uncertainty on the observations. This 
      defaults to zeros (i.e. the data are assumed to be known 
      perfectly).

    obs_diff : (D,) tuple, optional
      Derivative of the observations. For example, use (1,) if the 
      observations constrain the slope of a 1-D Gaussian process.
      
    Returns
    -------
    out : GaussianProcess
      
    '''
    y = np.asarray(y,dtype=float)
    d = np.asarray(d,dtype=float)
    q,dim = y.shape
    # if the GaussianProcess already has dim specified then make sure 
    # the data dim is the same
    if self.dim is not None:
      if self.dim != dim:
        raise ValueError(
          'The number of spatial dimensions for *y* is inconsistent '
          'with the Gaussian process.')

    if obs_diff is None:
      obs_diff = np.zeros(dim,dtype=int)
    else:
      obs_diff = np.asarray(obs_diff,dtype=int)
      if obs_diff.shape[0] != dim:
        raise ValueError(
          'The number of spatial dimensions for *obs_diff* is '
          'inconsistent with *y*')
    
    if sigma is None:
      sigma = np.zeros(q,dtype=float)      
    else:
      sigma = np.asarray(sigma,dtype=float)
    
    mean,covariance = _condition_factory(self,y,d,sigma,obs_diff)
    out = GaussianProcess(mean,covariance,dim=dim,order=-1)
    return out

  def recursive_condition(self,y,d,sigma=None,obs_diff=None,
                          max_chunk=None):                           
    ''' 
    Returns a conditional Gaussian process which incorporates the 
    observed data. The data is broken into chunks and the returned 
    *GaussianProcess* is computed recursively, where each recursion 
    depth corresponds to a different chunk. The *GaussianProcess* 
    returned by this function should be equivalent (to within 
    numerical precision) to the *GaussianProcess* returned by the 
    *condition* method. This function should be preferred over the 
    *condition* method for large (>1000) data sets because it can be 
    faster and use less memory.
    
    Parameters
    ----------
    y : (N,D) array
      Observation points
    
    d : (N,) array
      Observed values at *y*
      
    sigma : (N,) array, optional
      One standard deviation uncertainty on the observations. This 
      defaults to zeros (i.e. the data are assumed to be known 
      perfectly).

    obs_diff : (D,) tuple, optional
      Derivative of the observations. For example, use (1,) if the 
      observations constrain the slope of a 1-D Gaussian process.
      
    max_chunk : int, optional
      Maximum size of the data chunks. Defaults to *max(500,N/10)*.
      
    Returns
    -------
    out : GaussianProcess
      
    '''
    y = np.asarray(y,dtype=float)
    d = np.asarray(d,dtype=float)
    q = y.shape[0]
    if sigma is None:
      sigma = np.zeros(q,dtype=float)      
    else:
      sigma = np.asarray(sigma,dtype=float)

    if max_chunk is None:
      max_chunk = max(500,q//10)
    
    out = self    
    count = 0        
    while count < q: 
      idx = range(count,min(count+max_chunk,q))
      out = out.condition(y[idx],d[idx],sigma=sigma[idx],
                          obs_diff=obs_diff)
      count = idx[-1] + 1
      
    return out    
    

  def mean(self,x,diff=None,retry=1):
    ''' 
    Returns the mean of the Gaussian process 
    
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
    if diff is None:  
      diff = np.zeros(x.shape[1],dtype=int)
    else:
      diff = np.asarray(diff,dtype=int)
      
    if self.dim is not None:
      if x.shape[1] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *x* is inconsistent with '
          'the GaussianProcess.')

      if diff.shape[0] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *diff* is inconsistent with '
          'the GaussianProcess.')
      
    out = self._mean(x,diff)
    # If *out* is not finite then warn the user and attempt to compute 
    # it again. An error is raised after *retry* attempts.
    if not np.all(np.isfinite(out)):
      if retry > 0:
        warnings.warn(
          'Encountered non-finite value in the mean of the Gaussian '
          'process. This may be due to a CPU fluke. All relevant ' 
          'caches will be cleared and another attempt will be made to '
          'compute the mean.')
        self.clear_cache()  
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
    Returns the covariance of the Gaussian process 
    
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
    x2 = np.asarray(x2,dtype=float)
    if diff1 is None:
      diff1 = np.zeros(x1.shape[1],dtype=int)
    else:
      diff1 = np.asarray(diff1,dtype=int)

    if diff2 is None:  
      diff2 = np.zeros(x2.shape[1],dtype=int)
    else:
      diff2 = np.asarray(diff2,dtype=int)
      
    if self.dim is not None:
      if x1.shape[1] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *x1* is inconsistent '
          'with the GaussianProcess.')

      if x2.shape[1] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *x2* is inconsistent '
          'with the GaussianProcess.')

      if diff1.shape[0] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *diff1* is '
          'inconsistent with the GaussianProcess.')

      if diff2.shape[0] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *diff2* is '
          'inconsistent with the GaussianProcess.')

    out = self._covariance(x1,x2,diff1,diff2)
    # If *out* is not finite then warn the user and attempt to compute 
    # it again. An error is raised after *retry* attempts.
    if not np.all(np.isfinite(out)):
      if retry > 0:
        warnings.warn(
          'Encountered non-finite value in the covariance of the '
          'Gaussian process. This may be due to a CPU fluke. All ' 
          'relevant caches will be cleared and another attempt will '
          'be made to compute the covariance.')
        self.clear_cache()  
        return self.covariance(x1,x2,diff1=diff1,diff2=diff2,
                               retry=retry-1)
      else:    
        raise ValueError(
          'Encountered non-finite value in the covariance of the '
          'Gaussian process.')     

    # return a copy of *out* that is safe to write to
    out = np.array(out,copy=True)
    return out
    
  def mean_and_uncertainty(self,x,max_chunk=100):
    ''' 
    Returns the mean and uncertainty at *x*. This does not return the 
    full covariance matrix, making it appropriate for evaluating the 
    Gaussian process at many interpolation points.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    max_chunk : int, optional  
      Break *x* into chunks with this size and evaluate the Gaussian 
      process for each chunk. Smaller values result in decreased 
      memory usage but also decrease speed.
    
    Returns
    -------
    out_mean : (N,) array
      Mean of the Gaussian process at *x*.
    
    out_sigma : (N,) array  
      One standard deviation uncertainty of the Gaussian process at 
      *x*.
      
    '''
    count = 0
    x = np.asarray(x)
    q = x.shape[0]
    out_mean = np.zeros(q)
    out_sigma = np.zeros(q)
    while count < q:
      idx = range(count,min(count+max_chunk,q))
      out_mean[idx] = self.mean(x[idx])
      cov = self.covariance(x[idx],x[idx])
      var = np.diag(cov)
      out_sigma[idx] = np.sqrt(var)
      count = idx[-1] + 1
    
    return out_mean,out_sigma

  def draw_sample(self,x,tol=1e-10):  
    '''  
    Draws a random sample from the Gaussian process
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    Returns
    -------
    out : (N,) array      
    
    Notes
    -----
    This function does not check if the covariance function at *x* is 
    positive definite. If it is not, then the covariance function is 
    invalid and then the returned sample will be meaningless. If you 
    are not confident that the covariance function is positive 
    definite then call the *is_positive_definite* method with argument 
    *x*. 

    '''
    if self.order != -1:
      warnings.warn(
        'Cannot sample a *GaussianProcess* with a polynomial null '
        'space. The sample will instead be generated from a '
        'conditional *GaussianProcesss* where the null space has '
        'been removed.')

    mean = self.mean(x)
    cov = self.covariance(x,x)
    out = _draw_sample(mean,cov,tol=tol)
    return out
    
  def is_positive_definite(self,x,tol=1e-10):
    '''     
    Tests if the covariance matrix, which is the covariance function 
    evaluated at *x*, is positive definite by checking if all the 
    eigenvalues are real and positive. An affirmative result from this 
    test is necessary but insufficient to ensure that the covariance 
    function is positive definite.
    
    If this function returns a False then there was likely an 
    inappropriate choice for *basis* in the *PriorGaussianProcess*. 
    Perhaps the chosen basis is not sufficiently differentiable. 
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
    
    tol : float, optional
      A matrix which should be positive definite may still have 
      slightly negative or slightly imaginary eigenvalues because of 
      numerical rounding error. This arguments sets the tolerance for 
      negative or imaginary eigenvalues.

    Returns
    -------
    out : bool

    '''
    cov = self.covariance(x,x)    
    out = _is_positive_definite(cov,tol)
    return out  
    

  def clear_cache(self):
    ''' 
    Attempts to clear the caches of memoized functions associated with 
    this Gaussian process. This is done by calling the *clear_cache* 
    method of the mean and covariance function if one exists. Note 
    that this may clear the caches of other Gaussian processes which 
    have shared ancestors.
    '''
    _mvmonos.clear_cache()
    if hasattr(self._mean,'clear_cache'):
      self._mean.clear_cache()

    if hasattr(self._covariance,'clear_cache'):
      self._covariance.clear_cache()
          
    
class PriorGaussianProcess(GaussianProcess):
  ''' 
  A *PriorGaussianProcess* instance represents a stationary Gaussian 
  process process which has a constant mean and a covariance function 
  described by a radial basis function. The prior can also be given a 
  null space containing all polynomials of order *order*.  

  Parameters
  ----------
  basis : RBF instance
    Radial basis function describing the covariance function
    
  coeff : 3-tuple  
    Tuple of three distribution coefficients, *a*, *b*, and *c*. *a* 
    is the mean, *b* scales the amplitude of the covariance, and *c* 
    is the characteristic length scale (see above).
      
  order : int, optional
    Order of the polynomial spanning the null space. Defaults to -1, 
    which means that there is no null space.
    
  dim : int, optional
    Fixes the spatial dimensions of the Gaussian process.   
  
  Examples
  --------
  Instantiate a *PriorGaussianProcess* where the basis is a squared 
  exponential function with mean = 0, variance = 1, and characteristic 
  length scale = 2.
  
  >>> from rbf.basis import ga,phs3
  >>> gp = PriorGaussianProcess(ga,(0,1,2))
  
  Instantiate a PriorGaussianProcess which is equivalent to a 1-D thin 
  plate spline with penalty parameter 0.01. Then find the conditional 
  mean and uncertainty of the Gaussian process after incorporating 
  observations
  
  >>> gp = PriorGaussianProcess(phs3,(0.0,0.01,1.0),order=1)
  >>> y = np.array([[0.0],[0.5],[1.0],[1.5],[2.0]])
  >>> d = np.array([0.5,1.5,1.25,1.75,1.0])
  >>> sigma = np.array([0.1,0.1,0.1,0.1,0.1])
  >>> gp = gp.condition(y,d,sigma)
  >>> x = np.linspace(0.0,2.0,100)[:,None]
  >>> mean,sigma = gp(x)
  
  Notes
  -----
  If *order* >= 0, then *a* has no effect on the resulting Gaussian 
  process.
  
  If *basis* is scale invariant, such as for odd order polyharmonic 
  splines, then *b* and *c* have inverse effects on the resulting 
  Gaussian process and thus only one of them needs to be chosen while 
  the other can be fixed at an arbitary value.
  
  Not all radial basis functions are positive definite, which means 
  that there may not be a valid covariance function describing the 
  Gaussian process. The squared exponential basis function, 
  *rbf.basis.ga*, is positive definite for all spatial dimensions and 
  it is infinitely differentiable. For this reason it is a generally 
  safe choice for *basis*.

  '''
  def __init__(self,basis,coeff,order=-1,dim=None):
    coeff = np.asarray(coeff,dtype=float)  
    if coeff.shape[0] != 3:
      raise ValueError('*coeff* must be a (3,) array')
    if coeff[2] == 0.0:   
      raise ValueError('characteristic length-scale cannot be zero')
      
    mean,covariance = _prior_factory(basis,coeff,order)
    GaussianProcess.__init__(self,mean,covariance,
                             order=order,dim=dim)

    # A PriorGaussian process has these additional attributes
    self.basis = basis
    self.coeff = coeff
    self._repr_string = None
    
  def __repr__(self):
    # make the repr string once and then reuse it.
    if self._repr_string is None:
      # make string for __repr__
      a = _sigfigs(self.coeff[0],3)
      b = _sigfigs(self.coeff[1],3)
      c_inv = _sigfigs(1.0/self.coeff[2],3)
      eps = rbf.basis.get_eps()
      cov_expr = b*self.basis.expr.subs(eps,c_inv)
      try:
        # try to simplify cov_expr to a float. If its possible then try 
        # to convert NaNs to zeros
        cov_expr = np.float64(cov_expr)
        cov_expr = np.nan_to_num(cov_expr)
      except TypeError:  
        # Just use the expression
        pass
    
      self._repr_string = (
        '<PriorGaussianProcess : mean = %s, cov = %s, order = %s>' 
        % (a,str(cov_expr),self.order))

    return self._repr_string
    

def gpr(y,d,sigma,coeff,x=None,basis=rbf.basis.ga,order=1,
        diff=None,procs=0):
  '''     
  Performs Guassian process regression on the observed data. This is a 
  convenience function which initiates a *PriorGaussianProcess*, 
  conditions it with the observations, differentiates it (if 
  specified), and then evaluates the resulting *GaussianProcess* at 
  *x*. 
  
  Parameters
  ----------
  y : (N,D) array
    Observation points.

  d : (...,N) array
    Observed data at *y*.
  
  sigma : (...,N) array
    Data uncertainty. *np.nan* or *np.inf* can be used to indicate 
    that the data is missing, which will cause the corresponding value 
    in *d* to be ignored.
  
  coeff : 3-tuple
    Variance, mean, and characteristic length scale for the prior 
    Gaussian process.
  
  x : (M,D) array, optional
    Evaluation points, defaults to *y*.
  
  basis : RBF instance, optional      
    Radial basis function which describes the prior covariance 
    structure. Defaults to rbf.basis.ga.
    
  order : int, optional
    Order of the prior null space.

  diff : (D,), optional         
    Specifies the derivative of the returned values. 

  procs : int, optional
    Distribute the tasks among this many subprocesses. This defaults 
    to 0 (i.e. the parent process does all the work).  Each task is to 
    perform Gaussian process regression for one of the (N,) arrays in 
    *d* and *sigma*. So if *d* and *sigma* are (N,) arrays then using 
    multiple process will not provide any speed improvement.
  
  Returns
  -------
  out_mean : (...,M) array  
    Mean of the posterior at *x*.
      
  out_sigma : (...,M) array  
    One standard deviation of the posterior at *x*.
      
  '''
  y = np.asarray(y)
  d = np.asarray(d)
  sigma = np.asarray(sigma)
  if diff is None:   
    diff = np.zeros(y.shape[1],dtype=int)

  if x is None:
    x = y
      
  m = x.shape[0]
  bcast_shape = d.shape[:-1]
  q = int(np.prod(bcast_shape))
  n = y.shape[0]
  d = d.reshape((q,n))
  sigma = sigma.reshape((q,n))

  def doit(i):
    logger.debug('Performing GPR on data set %s of %s ...' % (i+1,q))
    gp = PriorGaussianProcess(basis,coeff,order=order)
    # do not condition the Gaussian process with data that has 
    # infinite or NaN uncertainty.
    is_finite = np.isfinite(sigma[i])
    gp = gp.recursive_condition(y[is_finite],d[i,is_finite],
                                sigma=sigma[i,is_finite])
    gp = gp.differentiate(diff)
    out_mean_i,out_sigma_i = gp.mean_and_uncertainty(x)
    return out_mean_i,out_sigma_i

  out = rbf.mp.parmap(doit,range(q),workers=procs)   
  out_mean = np.array([k[0] for k in out])
  out_sigma = np.array([k[1] for k in out])
  out_mean = out_mean.reshape(bcast_shape + (m,))
  out_sigma = out_sigma.reshape(bcast_shape + (m,))
  return out_mean,out_sigma

  

  
