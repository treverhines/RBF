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
from a new object oriented perspective. I treat GPR as a method of a 
*GaussianProcess* and it returns a new *GaussianProcess* which can 
itself be used as a prior for further GPR. *GaussianProcess* instances 
also have methods for addition, subtraction, scaling, and 
differentiation, which each return a *GaussianProcess* possessing the 
same methods. I find this object oriented approach for Gaussian 
processes to be extensible and it has allowed me to tackle a wider 
range of problems in my research which I could not have accomplished 
with existing software.

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
:math:`d_u=-1`, which is a programmatic and notational convenience. We 
express the Gaussian process as
  
.. math::
  u = u_o + \sum_{i=1}^{m_u} c_i p_i,

where :math:`\{c_i\}_{i=1}^{m_u}` are uncorrelated random variables 
with infinite variance and

.. math::
  u_o \\sim \\mathcal{N}\\big(\\bar{u},C_u\\big).

We endow the Gaussian process with five operations: addition, 
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
  u + v = z

where the mean, covariance, and null space order for :math:`z` are

.. math::
  \\bar{z} = \\bar{u} + \\bar{v},

.. math::
  C_z = C_u + C_v,
  
and 

.. math::
  d_z = \max(d_u,d_v).

Subtraction
-----------
A Gaussian process can be subtracted from another Gaussian processes 
as

.. math::
  u - v = z 

where 

.. math::
  \\bar{z} = \\bar{u} - \\bar{v},

.. math::
  C_z = C_u + C_v,
  
and 

.. math::
  d_z = \max(d_u,d_v).


Scaling
-------
A Gaussian process can be scaled by a constant as 

.. math::
  cu = z 

where 

.. math::
  \\bar{z} = c\\bar{u},

.. math::
  C_z = c^2C_u,

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
operator,

.. math::
  D = \\frac{\partial^{a_1 + a_2 + \dots + a_n}}
            {\partial x_1^{a_1} \partial x_2^{a_2} \dots 
            \partial x_n^{a_n}},

as 

.. math::
  Du = z 

where 

.. math::
  \\bar{z} = D\\bar{u},
  
.. math::
  C_z = DC_uD^H,
  
.. math::
  d_z = \max(d_u - d_D,-1),

and :math:`d_D = a_1 + a_2 + \dots + a_n`. In the expression for the 
covariance function, the differential operator is differentiating
:math:`C_u(x,x')` with respect to :math:`x`, and the adjoint 
differential operator, :math:`D^H`, is differentiating 
:math:`C_u(x,x')` with respect to :math:`x'`.

Conditioning
------------
A Gaussian process can be conditioned with :math:`q` noisy 
observations of :math:`u(x)`, :math:`\mathbf{d}=\{d_i\}_{i=1}^q`, 
which have been made at locations :math:`\mathbf{y}=\{y_i\}_{i=1}^q`. 
These observations have noise with zero mean and covariance described 
by :math:`\mathbf{C_d}`. The conditioned Gaussian process is 

.. math::
  u | \mathbf{d} = z 
  
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
                \mathbf{k}(x',\mathbf{y})^H,                

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
    [\mathbf{p}_u(y_i)]^H_{y_i \in \mathbf{y}}   
    & \mathbf{0}    \\\\
  \\end{array}  
  \\right].

We define the residual vector as

.. math::
  \mathbf{r} = \\left([d_i - \\bar{u}(y_i)]_{i=1}^q\\right)^H
  
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
We begin Gaussian process regression by assuming a prior Gaussian 
process which describes what we believe the underlying function looks 
like. In this module, the mean and covariance for prior Gaussian 
processes can be described as
  
.. math::
  \\bar{u}(x) = b,
  
and

.. math::
  C_u(x,x') = a\phi\\left(\\frac{||x - x'||_2}{c}\\right), 
  
where :math:`a`, :math:`b`, and :math:`c` are user specified 
coefficients. The literature on radial basis functions and Gaussian 
process regression often refers to :math:`c` as the shape parameter or 
the characteristic length scale. :math:`\phi` is a user specified 
positive definite radial function. One common choice for :math:`\phi` 
is the squared exponential function,

.. math::
  \phi(r) = \exp(-r^2),

which has the benefit of being infinitely differentiable. See [1] for 
an exhaustive list of positive definite radial functions.

References
==========
[1] Rasmussen, C., and Williams, C., Gaussian Processes for Machine 
Learning. The MIT Press, 2006.

'''
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
from functools import wraps
import warnings

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
  val,vec = np.linalg.eig(cov)
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


def _warn_if_null_space_exists(fin):
  @wraps(fin)
  def fout(self,*args,**kwargs):
    quiet = kwargs.pop('quiet',False)
    if (self.order != -1) & (not quiet): 
      warnings.warn(
        'The method *%s* has been called for a GaussianProcess with a '
        'polynomial null space. The output is for a conditional '
        'GaussianProcesss where the null space has been fixed at zero.' 
        % fin.__name__)

    return fin(self,*args,**kwargs)    

  return fout
  

class GaussianProcess(object):
  ''' 
  A *GaussianProcess* instance represents a stochastic process, which 
  is defined in terms of its mean function, covariance function, and 
  polynomial null space. This clas allows for basic operations on 
  Gaussian processes which includes addition, subtraction, scaling, 
  differentiation, sampling, and conditioning.
    
  This class does not check whether the specified covariance function 
  is positive definite, making it easy construct an invalid 
  *GaussianProcess* instance. For this reason, this class should not 
  be directly instantiated by the user. Instead, create a 
  *GaussianProcess* with the subclass *PriorGaussianProcess*.
    
  Parameters
  ----------
  mean_func : function
    Evaluates the mean function for the Gaussian process at *x*, 
    where *x* is a two-dimensional array of positions. This function 
    should also be able to return the spatial derivatives of the 
    mean function, which is specified with *diff*.  The positional 
    arguments for this function must be *x*, *diff*, and the 
    elements of *func_args*.

  cov_func : function
    Evaluates the covariance function for the Gaussian process at 
    *x1* and *x2*.  This function should also be able to return the 
    covariance of the spatial derivatives of *x1* and *x2*, which 
    are specified with *diff1* and *diff2*. The positional arguments 
    for this function must be *x1*, *x2*, *diff1*, *diff2*, and the 
    elements of *func_args*.

  func_args : tuple, optional
    Additional positional arguments passed to *mean_func* and 
    *cov_func*.
    
  order : int, optional
    Order of the polynomial null space. If this is -1 then the 
    Gaussian process contains no null space. This should be used if 
    the data contains trends that are well described by a polynomial.
    
  dim : int, optional  
    Specifies the spatial dimensions of the Gaussian process. An 
    error will be raised if the arguments to the *mean* or 
    *covariance* methods conflict with *dim*.
    
  '''
  def __init__(self,mean_func,cov_func,func_args=(),order=-1,dim=None):
    # these functions are hidden because *mean* and *covariance* 
    # should be preferred
    self._mean_func = mean_func 
    self._cov_func = cov_func
    self._func_args = func_args
    self.order = order
    self.dim = dim
  
  def __call__(self,*args,**kwargs):
    ''' 
    gp(*args,**kwargs) <==> gp.mean_and_uncertainty(*args,**kwargs)
    '''
    return self.mean_and_uncertainty(*args,**kwargs)

  def __add__(self,other):
    ''' 
    gp1 + gp2 <==> gp1.sum(gp2)
    '''
    return self.add(other)

  def __sub__(self,other):
    ''' 
    gp1 - gp2 <==> gp1.difference(gp2)
    '''
    return self.subtract(other)

  def __mul__(self,c):
    ''' 
    c*gp  <==> gp.scale(c)
    '''
    return self.scale(c)

  def __rmul__(self,c):
    ''' 
    gp*c  <==> gp.scale(c)
    '''
    return self.__mul__(c)

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
    def mean_func(x,diff):
      out = self.mean(x,diff=diff) + other.mean(x,diff=diff)
      return out       

    def cov_func(x1,x2,diff1,diff2):
      out = (self.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             other.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out
            
    order = max(self.order,other.order)
    out = GaussianProcess(mean_func,cov_func,order=order)
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
    def mean_func(x,diff):
      out = self.mean(x,diff=diff) - other.mean(x,diff=diff)
      return out
      
    def cov_func(x1,x2,diff1,diff2):
      out = (self.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             other.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out       
            
    order = max(self.order,other.order)
    out = GaussianProcess(mean_func,cov_func,order=order)
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
    def mean_func(x,diff):
      out = c*self.mean(x,diff=diff)
      return out

    def cov_func(x1,x2,diff1,diff2):
      out = c**2*self.covariance(x1,x2,diff1=diff1,diff2=diff2)
      return out
      
    if c != 0.0:
      order = self.order
    else:
      order = -1
        
    out = GaussianProcess(mean_func,cov_func,order=order)
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
    dim = len(d)
    d = np.asarray(d,dtype=int)
    def mean_func(x,diff):
      out = self.mean(x,diff=diff+d)
      return out 

    def cov_func(x1,x2,diff1,diff2):
      out = self.covariance(x1,x2,
                            diff1=diff1+d,
                            diff2=diff2+d)
      return out
      
    order = max(self.order - sum(d),-1)
    out = GaussianProcess(mean_func,cov_func,dim=dim,order=order)
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
    y = np.asarray(y)
    d = np.asarray(d)
    q,dim = y.shape
    if obs_diff is None:
      obs_diff = np.zeros(dim,dtype=int)
    else:
      obs_diff = np.asarray(obs_diff,dtype=int)
    
    if sigma is None:
      sigma = np.zeros(q)      
    else:
      sigma = np.asarray(sigma)

    powers = rbf.poly.powers(self.order,dim) 
    m = powers.shape[0]
    Cu_yy = self.covariance(y,y,diff1=obs_diff,diff2=obs_diff)
    Cd = np.diag(sigma**2)
    p_y = rbf.poly.mvmonos(y,powers,diff=obs_diff)
    K_y = np.zeros((q+m,q+m))
    K_y[:q,:q] = Cu_yy + Cd
    K_y[:q,q:] = p_y
    K_y[q:,:q] = p_y.T
    try:
      K_y_inv = np.linalg.inv(K_y)
    except np.linalg.LinAlgError:
      raise np.linalg.LinAlgError(
          'Failed to compute the inverse of K. This is likely '
          'because there is not enough data to constrain a null '
          'space in the prior')

    # compute residuals
    r = np.zeros(q+m)
    r[:q] = d - self.mean(y,diff=obs_diff)
    
    def mean_func(x,diff):
      Cu_xy = self.covariance(x,y,diff1=diff,diff2=obs_diff)
      p_x   = rbf.poly.mvmonos(x,powers,diff=diff)
      k_xy  = np.hstack((Cu_xy,p_x))
      out = self.mean(x,diff=diff) + k_xy.dot(K_y_inv.dot(r))
      return out

    def cov_func(x1,x2,diff1,diff2):
      Cu_x1x2 = self.covariance(x1,x2,diff1=diff1,diff2=diff2)
      Cu_x1y  = self.covariance(x1,y,diff1=diff1,diff2=obs_diff)
      Cu_x2y  = self.covariance(x2,y,diff1=diff2,diff2=obs_diff)
      p_x1  = rbf.poly.mvmonos(x1,powers,diff=diff1)
      p_x2  = rbf.poly.mvmonos(x2,powers,diff=diff2)
      k_x1y = np.hstack((Cu_x1y,p_x1))
      k_x2y = np.hstack((Cu_x2y,p_x2))
      out = Cu_x1x2 - k_x1y.dot(K_y_inv).dot(k_x2y.T) 
      return out

    out = GaussianProcess(mean_func,cov_func,dim=dim,order=-1)
    return out

  def mean(self,x,diff=None):
    ''' 
    Returns the mean of the Gaussian process 
    
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
    x = np.asarray(x)
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
      
    out = self._mean_func(x,diff,*self._func_args)
    return out

  def covariance(self,x1,x2,diff1=None,diff2=None):
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

    Returns
    -------
    out : (N,N) array    
    
    '''
    x1 = np.asarray(x1) 
    x2 = np.asarray(x2) 
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
          'The number of spatial dimensions for *x1* is inconsistent with '
          'the GaussianProcess.')

      if x2.shape[1] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *x2* is inconsistent with '
          'the GaussianProcess.')

      if diff1.shape[0] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *diff1* is inconsistent with '
          'the GaussianProcess.')

      if diff2.shape[0] != self.dim:
        raise ValueError(
          'The number of spatial dimensions for *diff2* is inconsistent with '
          'the GaussianProcess.')

    out = self._cov_func(x1,x2,diff1,diff2,*self._func_args)
    return out
    
  def mean_and_uncertainty(self,x,diff=None,max_chunk=100):
    ''' 
    Returns the mean and uncertainty at *x*. This does not return the 
    full covariance matrix, making it appropriate for evaluating the 
    Gaussian process at many interpolation points.
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    diff : (D,) tuple, optional
      Derivative specification
      
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
      out_mean[idx] = self.mean(x[idx],diff=diff)
      cov = self.covariance(x[idx],x[idx],diff1=diff,diff2=diff)
      out_sigma[idx] = np.sqrt(np.diag(cov))
      count = idx[-1] + 1
    
    return out_mean,out_sigma

  def draw_sample(self,x,diff=None,tol=1e-10):  
    '''  
    Draws a random sample from the Gaussian process
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    diff : (D,) tuple
      Derivative specification
      
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

    mean = self.mean(x,diff=diff)
    cov = self.covariance(x,x,diff1=diff,diff2=diff)
    return _draw_sample(mean,cov,tol=tol)
    
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
    scales the variance of the Gaussian process, *b* is the mean, 
    and *c* is the characteristic length scale (see above). 
      
  order : int, optional
    Order of the polynomial spanning the null space. Defaults to -1, 
    which means that there is no null space.
    
  dim : int, optional
    Fixes the spatial dimensions of the Gaussian process.   
  
  Examples
  --------
  Instantiate a PriorGaussianProcess where the basis is a squared 
  exponential function with variance = 1, mean = 0, and characteristic 
  length scale = 2.
  
  >>> from rbf.basis import ga
  >>> gp = PriorGaussianProcess(ga,(1,0,2))
  
  Instantiate a PriorGaussianProcess which is equivalent to a 1-D thin 
  plate spline with penalty parameter 0.01. Then find the conditional 
  mean and covariance of the Gaussian process after incorporating 
  observations
  
  >>> gp = rbf.gpr.PriorGaussianProcess(phs3,(0.01,0,1.0),order=1)
  >>> y = np.array([[0.0],[0.5],[1.0],[1.5],[2.0]])
  >>> d = np.array([0.5,1.5,1.25,1.75,1.0])
  >>> sigma = np.array([0.1,0.1,0.1,0.1,0.1])
  >>> gp = gp.condition(y,d,sigma)
  >>> x_interp = np.linspace(0.0,2.0,100)[:,None]
  >>> mean,cov = gp(x_interp)
  
  Notes
  -----
  If *order* >= 0, then *b* has no effect on the resulting Gaussian 
  process.
  
  If *basis* is scale invariant, such as for odd order polyharmonic 
  splines, then *a* and *c* have inverse effects on the resulting 
  Gaussian process and thus only one of them needs to be chosen while 
  the other can be fixed at an arbitary value.
  
  Not all radial basis functions are positive definite, which means 
  that there may not be a valid covariance function describing the 
  Gaussian process. The squared exponential basis function, 
  rbf.basis.exp, is positive definite for all spatial dimensions and 
  it is infinitely differentiable. For this reason it is a generally 
  safe choice for *basis*.

  '''
  def __init__(self,basis,coeff,order=-1,dim=None):
    def mean_func(x,diff):
      if sum(diff) == 0:
        out = coeff[1]*np.ones(x.shape[0])
      else:  
        out = np.zeros(x.shape[0])

      return out
      
    def cov_func(x1,x2,diff1,diff2):
      eps = np.ones(x2.shape[0])/coeff[2]
      a = (-1)**sum(diff2)*coeff[0]
      diff = diff1 + diff2
      out = a*basis(x1,x2,eps=eps,diff=diff)
      if np.any(~np.isfinite(out)):
        raise ValueError(
          'Encountered a non-finite covariance. This is likely '
          'because the prior basis function is not sufficiently '
          'differentiable.')

      return out
      
    GaussianProcess.__init__(self,mean_func,cov_func,
                             order=order,dim=dim)

