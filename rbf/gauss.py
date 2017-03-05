''' 
This module defines a class, *GaussianProcess*, which is an 
abstraction that allows one to easily work with Gaussian processes. 
The *GaussianProcess* class is primarily intended for Gaussian process 
regression (GPR), which is performed with the *condition* method. GPR 
is a technique for constructing a continuous function from discrete 
and potentially noisy observations. This documentation describes 
Gaussian processes and the operations (methods), which they are 
endowed with. Details on the classes *GaussianProcess* and 
*RBFGaussianProcess* can be found in their doc strings.

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
  C_u(x,x') = b\phi\\left(||x - x'||_2,c\\right), 
  
where :math:`a`, :math:`b`, and :math:`c` are user specified 
coefficients. The literature on radial basis functions and Gaussian 
process regression often refers to :math:`c` as the shape parameter or 
the characteristic length scale. :math:`\phi` is a user specified 
positive definite radial function. One common choice for :math:`\phi` 
is the squared exponential function,

.. math::
  \phi(r,c) = \exp\\left(-r^2/c^2\\right),

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
import logging
import weakref
import inspect
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


def _draw_sample(mean,cov):
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
    return np.float64(0.0)
  
  if ~np.isfinite(val):
    return np.float64(val)
    
  d = -np.int(np.log10(np.abs(val))) + n - 1 
  out = np.round(val,d)
  return out
    

class Memoize(object):
  ''' 
  Memoizing decorator. The output for calls to decorated functions 
  will be cached and reused if the function is called again with the 
  same arguments. The input arguments must either be hashable or numpy 
  arrays. Caches can be cleared with the module-level function 
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
    # generates hashable representations of the arguments
    def hashables():
      for a in args:
        if hasattr(a,'tobytes'):
          # if *a* has the method *tobytes*, then it is a numpy array. 
          # The output for *tobytes* is used in the key, since the array 
          # is not hashable.
          yield a.tobytes()
        else:
          yield a
          
    # create the cache key. 
    key = tuple(hashables())        
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


@Memoize
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
  @Memoize
  def mean(x,diff):
    out = gp1._mean(x,diff) + gp2._mean(x,diff)
    return out       

  @Memoize
  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out

  @Memoize
  def null(x,diff):
    out = np.hstack((gp1._null(x,diff),
                     gp2._null(x,diff)))
    return out                     
            
  out = GaussianProcess(mean,covariance,null)
  return out
  

def _subtract_factory(gp1,gp2):
  '''   
  Factory function which returns the mean and covariance functions for 
  a *GaussianProcess* which has been subtracted from another 
  *GaussianProcess*.
  '''
  @Memoize
  def mean(x,diff):
    out = gp1._mean(x,diff) - gp2._mean(x,diff)
    return out
      
  @Memoize
  def covariance(x1,x2,diff1,diff2):
    out = (gp1._covariance(x1,x2,diff1,diff2) + 
           gp2._covariance(x1,x2,diff1,diff2))
    return out       
            
  @Memoize
  def null(x,diff):
    out = np.hstack((gp1._null(x,diff),
                     gp2._null(x,diff)))
    return out                     

  out = GaussianProcess(mean,covariance,null)
  return out


def _scale_factory(gp,c):
  '''   
  Factory function which returns the mean and covariance functions for 
  a scaled *GaussianProcess*.
  '''
  @Memoize
  def mean(x,diff):
    out = c*gp._mean(x,diff)
    return out

  @Memoize
  def covariance(x1,x2,diff1,diff2):
    out = c**2*gp._covariance(x1,x2,diff1,diff2)
    return out
      
  # the null space is unchanged by scaling
  out = GaussianProcess(mean,covariance,gp._null)
  return out


def _differentiate_factory(gp,d):
  '''   
  Factory function which returns the mean and covariance functions for 
  a differentiated *GaussianProcess*.
  '''
  @Memoize
  def mean(x,diff):
    out = gp._mean(x,diff + d)
    return out 

  @Memoize
  def covariance(x1,x2,diff1,diff2):
    out = gp._covariance(x1,x2,diff1+d,diff2+d)
    return out
      
  @Memoize
  def null(x,diff):
    out = gp._null(x,diff + d)
    return out 
    
  out = GaussianProcess(mean,covariance,null,dim=len(d))
  return out


def _condition_factory(gp,y,d,sigma,obs_diff):
  '''   
  Factory function which returns the mean and covariance functions for 
  a conditioned *GaussianProcess*.
  '''
  @Memoize
  def precompute():
    ''' 
    do as many calculations as possible without yet knowning where the 
    interpolation points will be.
    '''
    # compute K_y_inv
    Cd = np.diag(sigma**2)
    Cu_yy = gp._covariance(y,y,obs_diff,obs_diff)
    p_y   = gp._null(y,obs_diff)
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
    
  @Memoize
  def mean(x,diff):
    K_y_inv,r = precompute()
    Cu_xy = gp._covariance(x,y,diff,obs_diff)
    p_x   = gp._null(x,diff)
    k_xy  = np.hstack((Cu_xy,p_x))
    out   = gp._mean(x,diff) + k_xy.dot(K_y_inv.dot(r))
    return out

  @Memoize
  def covariance(x1,x2,diff1,diff2):
    K_y_inv,r = precompute()
    Cu_x1x2 = gp._covariance(x1,x2,diff1,diff2)
    Cu_x1y  = gp._covariance(x1,y,diff1,obs_diff)
    Cu_x2y  = gp._covariance(x2,y,diff2,obs_diff)
    p_x1    = gp._null(x1,diff1)
    p_x2    = gp._null(x2,diff2)
    k_x1y   = np.hstack((Cu_x1y,p_x1))
    k_x2y   = np.hstack((Cu_x2y,p_x2))
    out = Cu_x1x2 - k_x1y.dot(K_y_inv).dot(k_x2y.T) 
    return out
  
  @Memoize
  def null(x,diff):
    return np.empty((x.shape[0],0),dtype=float)

  out = GaussianProcess(mean,covariance,null)
  return out


def _prior_factory(basis,coeff):
  ''' 
  Factory function which returns the mean and covariance functions for 
  a *RBFGaussianProcess*.
  '''
  @Memoize
  def mean(x,diff):
    a,b,c = coeff  
    if sum(diff) == 0:
      out = np.full(x.shape[0],a,dtype=float)
    else:
      out = np.zeros(x.shape[0],dtype=float)

    return out
      
  @Memoize
  def covariance(x1,x2,diff1,diff2):
    a,b,c = coeff  
    diff = diff1 + diff2
    out = b*(-1)**sum(diff2)*basis(x1,x2,eps=c,diff=diff)
    if np.any(~np.isfinite(out)):
      raise ValueError(
        'Encountered a non-finite prior covariance. This may be '
        'because the prior basis function is not sufficiently '
        'differentiable.')

    return out

  @Memoize
  def null(x,diff):
    return np.empty((x.shape[0],0),dtype=float)
    
  return mean,covariance,null

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
  

class GaussianProcess(object):
  ''' 
  A *GaussianProcess* instance represents a stochastic process, which 
  is defined in terms of its mean function, covariance function, and 
  polynomial null space. This class allows for basic operations on 
  Gaussian processes which includes addition, subtraction, scaling, 
  differentiation, sampling, and conditioning.
    
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

  null : function, optional
    Null space basis functions. If two points in function space differ 
    only by a linear combination of the null space basis functions 
    then they have the same likelihood of being realized by this 
    Gaussian process.  This function takes either one argument, *x*, 
    or two arguments, *x* and *diff*. *x* is an (N,D) array of 
    positions and *diff* is a (D,) array specifying the derivative. 
    This function returns an (N,P) array, where each column is a basis 
    function spanning the null space evaluated at *x*. By default, the 
    a *GaussianProcess* instance contains no null space.
        
  dim : int, optional  
    Specifies the spatial dimensions of the Gaussian process. An error 
    will be raised if the arguments to the *mean* or *covariance* 
    methods conflict with *dim*.
    
  Notes
  -----
  1. This class does not check whether the specified covariance 
  function is positive definite, making it easy construct an invalid 
  *GaussianProcess* instance. For this reason, one may prefer to 
  create a *GaussianProcess* with the subclass *RBFGaussianProcess*.
  
  2. A *GaussianProcess* returned by *add*, *subtract*, *scale*, 
  *differentiate*, and *condition* has a *mean* and *covariance* 
  function which calls the *mean* and *covariance* functions of its 
  parents. For example, if *gp1* and *gp2* are *GaussianProcess* 
  instances then *gp_sum = gp1 + gp2* is another *GaussianProcess* 
  whose *mean* and *covariance* functions make calls to the *mean* and 
  *covariance* functions from its parents, *gp1* and *gp2*. Due to 
  this recursive implementation, the number of generations of children 
  (for lack of a better term) is limited by the maximum recursion 
  depth.

  3. If a Gaussian process contains a polynomial null space, then its 
  mean and covariance are undefined. This is because the coefficients 
  for the monomials spanning the null space are equally likely to be 
  any number between positive and negative infinity. When the *mean* 
  or *covariance* methods are called, the returned values are for the 
  Gaussian process under the condition that the monomial coefficients 
  are zero. In other words, the *mean* and *covariance* functions 
  ignore the presence of a polynomial null space.
  
  '''
  def __init__(self,mean,covariance,null=None,order=None,dim=None):
    if _get_arg_count(mean) == 1:
      # if the mean function only takes one argument then make a 
      # wrapper for it which takes two arguments.
      def mean_with_diff(x,diff):
        if sum(diff) != 0: 
          raise ValueError(
            'The mean of the Gaussian process is not differentiable')
          
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
            'The covariance of the Gaussian process is not '
            'differentiable')
          
        return covariance(x1,x2)

      self._covariance = covariance_with_diff
    else:
      self._covariance = covariance
    
    if null is None:  
      # Make an empty null space if one was not specified
      def null(x,diff):
        return np.empty((x.shape[0],0),dtype=float)
    
    if _get_arg_count(null) == 1:
      # if the null function only takes one argument then make a 
      # wrapper for it which takes two arguments.
      def null_with_diff(x,diff):
        if sum(diff) != 0: 
          raise ValueError(
            'The null space basis functions for the Gaussian process '
            'are not differentiable')
          
        return null(x)
    
      self._null = null_with_diff
    else:
      # otherwise, assume that the function can take two arguments
      self._null = null
        
    self.dim = dim
  
  def __call__(self,*args,**kwargs):
    ''' 
    equivalent to calling *mean_and_sigma*
    '''
    return self.mean_and_sigma(*args,**kwargs)

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
        
    out = GaussianProcess(
    mean,covariance,null = _add_factory(self,other)
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
    
    if d.ndim != 1:
      raise ValueError(
        'The observations, *d*, must be a one dimensional array')

    if sigma.ndim != 1:
      raise ValueError(
        'The observation uncertainties, *sigma*, must be a one '
        'dimensional array')
        
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
    returned by this method should be equivalent (to within numerical 
    precision) to the *GaussianProcess* returned by the *condition* 
    method. However, this methods run time, memory usaged, and 
    numerical stability may differ from the *condition* method.
    
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
    while True:
      idx = range(count,min(count+max_chunk,q))
      out = out.condition(y[idx],d[idx],sigma=sigma[idx],
                          obs_diff=obs_diff)
      count = min(count+max_chunk,q)
      if count == q:
        break
      
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
    
  def mean_and_sigma(self,x,max_chunk=100):
    ''' 
    Returns the mean and standard deviation at *x*. This does not 
    return the full covariance matrix, making it appropriate for 
    evaluating the Gaussian process at many points.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    max_chunk : int, optional  
      Break *x* into chunks with this size and evaluate the Gaussian 
      process for each chunk. This argument affects the speed and 
      memory usage of this method, and it does not affect the output. 
      Setting this to a larger value will reduce the number of python 
      function call at the expense of increased memory usage.
    
    Returns
    -------
    out_mean : (N,) array
      Mean of the Gaussian process at *x*.
    
    out_sigma : (N,) array  
      One standard deviation uncertainty of the Gaussian process at 
      *x*.
      
    '''
    count = 0
    x = np.asarray(x,dtype=float)
    q = x.shape[0]
    out_mean = np.zeros(q,dtype=float)
    out_sigma = np.zeros(q,dtype=float)
    # If q is zero then mean and covariance never get evaluated. This 
    # is a bug because errors can pass through
    while True:
      idx = range(count,min(count+max_chunk,q))
      out_mean[idx] = self.mean(x[idx])
      cov = self.covariance(x[idx],x[idx])
      var = np.diag(cov)
      out_sigma[idx] = np.sqrt(var)
      count = min(count+max_chunk,q)
      if count == q:
        break
    
    return out_mean,out_sigma

  def draw_sample(self,x):  
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
    mean = self.mean(x)
    cov = self.covariance(x,x)
    out = _draw_sample(mean,cov)
    return out
    
  def is_positive_definite(self,x,tol=1e-10):
    '''     
    Tests if the covariance matrix, which is the covariance function 
    evaluated at *x*, is positive definite by checking if all the 
    eigenvalues are real and positive. An affirmative result from this 
    test is necessary but insufficient to ensure that the covariance 
    function is positive definite.
    
    If this function returns a False then there was likely an 
    inappropriate choice for *basis* in the *RBFGaussianProcess*. 
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
    

class RBFGaussianProcess(GaussianProcess):
  ''' 
  A *RBFGaussianProcess* instance represents a stationary Gaussian 
  process which has a constant mean and a covariance function 
  described by a radial basis function. This can also be given a null 
  space containing all polynomials of order *order*.

  Parameters
  ----------
  coeff : 3-tuple  
    Tuple of three coefficients, *a*, *b*, and *c*, describing the 
    prior probability distribution.  *a* is the mean and, when using 
    the default value for *basis*, *b* and *c* describe the prior 
    variance and characteristic length-scale.  In general, *b* scales 
    the covariance function, and *c* is the shape parameter, *eps*, 
    that is used to define *basis*.
      
  basis : RBF instance, optional
    Radial basis function describing the covariance function. Defaults 
    to a squared exponential, *rbf.basis.se*.
    
  order : int, optional
    Order of the polynomial null space. Defaults to -1, which means 
    that there is no null space.
    
  dim : int, optional
    Fixes the spatial dimensions of the Gaussian process.   
  
  Examples
  --------
  Instantiate a *RBFGaussianProcess* where the basis is a squared 
  exponential function with mean = 0, variance = 1, and characteristic 
  length scale = 2.
  
  >>> gp = RBFGaussianProcess((0.0,1.0,2.0))
  
  Instantiate a RBFGaussianProcess which is equivalent to a 1-D thin 
  plate spline with penalty parameter 0.01. Then find the conditional 
  mean and uncertainty of the Gaussian process after incorporating 
  observations
  
  >>> from rbf.basis import phs3
  >>> gp = RBFGaussianProcess((0.0,0.01,1.0),basis=phs3,order=1)
  >>> y = np.array([[0.0],[0.5],[1.0],[1.5],[2.0]])
  >>> d = np.array([0.5,1.5,1.25,1.75,1.0])
  >>> sigma = np.array([0.1,0.1,0.1,0.1,0.1])
  >>> gp = gp.condition(y,d,sigma)
  >>> x = np.linspace(0.0,2.0,100)[:,None]
  >>> mean,sigma = gp(x)
  
  Notes
  -----
  1. If *order* >= 0, then *a* has no effect on the resulting Gaussian 
  process.
  
  2. If *basis* is scale invariant, such as for odd order polyharmonic 
  splines, then *b* and *c* have inverse effects on the resulting 
  Gaussian process and thus only one of them needs to be chosen while 
  the other can be fixed at an arbitary value.
  
  3. Not all radial basis functions are positive definite, which means 
  that it is possible to instantiate a *RBFGaussianProcess* that 
  does not have a valid covariance function. The squared exponential 
  basis function, *rbf.basis.se*, is positive definite for all spatial 
  dimensions. Furthermore, it is infinitely differentiable, which 
  means its derivatives are also positive definite. For this reason 
  *rbf.basis.se* is a generally safe choice for *basis*.

  4. See the notes in the *GaussianProcess* docstring.
  
  '''
  def __init__(self,coeff,basis=rbf.basis.se,order=-1,dim=None):
    coeff = np.asarray(coeff,dtype=float)  
    if coeff.shape[0] != 3:
      raise ValueError('*coeff* must be a (3,) array')
      
    mean,covariance = _prior_factory(basis,coeff,order)
    GaussianProcess.__init__(self,mean,covariance,order=order,dim=dim)
    # A RBFGaussian process has these additional private attributes
    self._basis = basis
    self._coeff = coeff
    
  def __repr__(self):
    # make the repr string once and then reuse it.
    if not hasattr(self,'_repr_string'):
      # make string for __repr__
      a = _sigfigs(self._coeff[0],3)
      b = _sigfigs(self._coeff[1],3)
      c = 1.0/_sigfigs(1.0/self._coeff[2],3)
      eps = rbf.basis.get_eps()
      cov_expr = b*self._basis.expr.subs(eps,c)
      try:
        # try to simplify cov_expr to a float. If this is possible, 
        # then convert NaNs to 0.0 which accounts for the singularity 
        # with PHS RBFs
        cov_expr = float(cov_expr)
        if np.isnan(cov_expr):
          cov_expr = 0.0
          
      except TypeError:  
        # Just use the expression
        pass
    
      self._repr_string = (
        '<RBFGaussianProcess : mean = %s, cov = %s, order = %s>' 
        % (a,str(cov_expr),self.order))

    return self._repr_string
    

# create alternate name
class PriorGaussianProcess(RBFGaussianProcess):
  def __init__(*args,**kwargs):
    print('*PriorGaussianProcess* has been renamed *RBFGaussianProcess*')
    RBFGaussianProcess.__init__(*args,**kwargs)
  
