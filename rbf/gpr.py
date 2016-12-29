''' 
Module for Gaussian process regression.
'''
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
from functools import wraps
import warnings

def _test_positive_definite(A,tol=1e-10):
  val,vec = np.linalg.eig(A)
  if np.any(val.real < -tol):
    return False

  if np.any(np.abs(val.imag) > tol):
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
  # make sure that all the eigenvalues are real and positive to within 
  # tolerance
  if np.any(val.real < -tol):
    raise ValueError('covariance matrix is not positive definite')
  if np.any(np.abs(val.imag) > tol):
    raise ValueError('covariance matrix is not positive definite')

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
    if (self._order != -1) & (not quiet): 
      warnings.warn(
        'The method *%s* has been called for a GaussianProcess with a '
        'polynomial null space. The output is for a conditional '
        'GaussianProcesss where the null space has been fixed at zero.' 
        % fin.__name__)

    return fin(self,*args,**kwargs)    

  return fout
  

class GaussianProcess(object):
  ''' 
  A *GaussianProcess* instance represents a stochastic process which 
  is defined in terms of its mean and covariance function. This class 
  allows for basic operations on Gaussian processes which includes 
  addition, subtraction, scaling, differentiation, sampling, and 
  conditioning.
    
  This class does not check whether the specified covariance function 
  is positive definite, making it easy construct an invalid 
  GaussianProcess instance. For this reason, this class should not be 
  directly instantiated by the user.  Instead, create a 
  GaussianProcess with the subclass *PriorGaussianProcess*.
    
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
    Order of the polynomial which spans the null space of the 
    Gaussian process. If this is -1 then the Gaussian process 
    contains no null space. If this is 0 then the likelihood of a 
    realization is unchanged by adding a constant. If this is 1 then 
    the likelihood of a realization is unchanged by adding a 
    constant and linear term, etc. This should be used if the data 
    contains trends that are well described by a polynomial.
    
  dim : int, optional  
    Specifies the spatial dimensions of the Gaussian process. An 
    error will be raised if the arguments to the *mean* or 
    *covariance* methods conflict with *dim*.
    
  '''
  def __init__(self,mean_func,cov_func,func_args=(),order=-1,dim=None):
    self._mean_func = mean_func
    self._cov_func = cov_func
    self._func_args = func_args
    self._order = order
    self._dim = dim
  
  def __call__(self,x,diff=None):
    ''' 
    Returns the mean and covariance evaluated at *x*
    
    Parameters
    ----------
    x : (N,D) array
      Evaluation points
      
    diff : (D,) tuple, optional
      Derivative specification
    
    Returns
    -------
    mean : (N,) array
    
    cov : (N,N) array  
      
    '''
    mean = self.mean(x,diff=diff)
    cov = self.covariance(x,x,diff1=diff,diff2=diff)
    return mean,cov

  def __add__(self,other):
    ''' 
    gp1 + gp2 <==> gp1.sum(gp2)
    '''
    return self.sum(other)

  def __sub__(self,other):
    ''' 
    gp1 - gp2 <==> gp1.difference(gp2)
    '''
    return self.difference(other)

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

  def sum(self,other):
    ''' 
    Adds two Gaussian processes
    
    Parameters
    ----------
    other : GuassianProcess 
      
    Returns
    -------
    out : GaussianProcess 

    Notes
    -----
    The order for the null space in the resulting Gaussian process is 
    set to the larger of the two null space orders for the input 
    Gaussian processes.

    '''
    def mean_func(x,diff):
      out = self.mean(x,diff=diff) + other.mean(x,diff=diff)
      return out       

    def cov_func(x1,x2,diff1,diff2):
      out = (self.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             other.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out
            
    order = max(self._order,other._order)
    out = GaussianProcess(mean_func,cov_func,order=order)
    return out

  def difference(self,other):
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
            
    order = max(self._order,other._order)
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
      
    order = self._order
    out = GaussianProcess(mean_func,cov_func,order=order)
    return out

  def derivative(self,diff):
    ''' 
    Returns the derivative of a Gaussian process
    
    Parameters
    ----------
    diff : (D,) tuple
      Derivative specification
      
    Returns
    -------
    out : GaussianProcess       

    '''
    dim = len(diff)
    diff_ = np.asarray(diff,dtype=int)
    def mean_func(x,diff):
      out = self.mean(x,diff=diff+diff_)
      return out 

    def cov_func(x1,x2,diff1,diff2):
      out = self.covariance(x1,x2,
                            diff1=diff1+diff_,
                            diff2=diff2+diff_)
      return out
      
    order = max(self._order - sum(diff_),-1)
    out = GaussianProcess(mean_func,cov_func,dim=dim,order=order)
    return out  

  def posterior(self,x,mu,sigma=None,diff=None):
    ''' 
    Returns a conditional Gaussian process which incorporates the 
    observed data.
    
    Parameters
    ----------
    x : (N,D) array
      Observation points
    
    mu : (N,) array
      Mean value of the observations  
      
    sigma : (N,) array, optional
      Standard deviation of the observations. This defaults to zeros 
      (i.e. the data are assumed to be known perfectly).

    diff : (D,) tuple, optional
      Derivative of the observations. For example, use (1,) if the 
      observations constrain the slope of a 1-D Gaussian process.
      
    Returns
    -------
    out : GaussianProcess
      
    '''
    x_ = np.asarray(x)
    mu = np.asarray(mu)
    n,dim = x_.shape
    if diff is None:
      diff_ = np.zeros(dim,dtype=int)
    else:
      diff_ = np.asarray(diff,dtype=int)
    
    if sigma is None:
      sigma = np.zeros(n)      
    else:
      sigma = np.asarray(sigma)

    powers = rbf.poly.powers(self._order,dim) 
    p = powers.shape[0]
    K = self.covariance(x_,x_,diff1=diff_,diff2=diff_)
    H = rbf.poly.mvmonos(x_,powers,diff=diff_)
    Cd = np.diag(sigma**2)
    A = np.zeros((n+p,n+p))
    A[:n,:n] = K + Cd
    A[:n,n:] = H
    A[n:,:n] = H.T
    try:
      Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
      raise np.linalg.LinAlgError(
          'Failed to compute the inverse covariance matrix. This is '
          'likely because there is not enough data to constrain a '
          'null space in the prior')

    # compute residuals
    res = np.zeros(n+p)
    res[:n] = mu - self.mean(x_,diff=diff_)
    
    def mean_func(x,diff):
      Ki = self.covariance(x,x_,diff1=diff,diff2=diff_)
      Hi = rbf.poly.mvmonos(x,powers,diff=diff)
      Ai = np.hstack((Ki,Hi))
      out = self.mean(x,diff=diff) + Ai.dot(Ainv.dot(res))
      return out

    def cov_func(x1,x2,diff1,diff2):
      Kii = self.covariance(x1,x2,diff1=diff1,diff2=diff2)
      Ki  = self.covariance(x1,x_,diff1=diff1,diff2=diff_)
      Kj  = self.covariance(x2,x_,diff1=diff2,diff2=diff_)
      Hi = rbf.poly.mvmonos(x1,powers,diff=diff1)
      Hj = rbf.poly.mvmonos(x2,powers,diff=diff2)
      Ai = np.hstack((Ki,Hi))
      Aj = np.hstack((Kj,Hj))
      out = Kii - Ai.dot(Ainv).dot(Aj.T) 
      return out

    out = GaussianProcess(mean_func,cov_func,dim=dim,order=-1)
    return out

  # convert _mean_func to a class method
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

    if self._dim is not None:
      if x.shape[1] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *x* is inconsistent with '
          'the GaussianProcess.')

      if diff.shape[0] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *diff* is inconsistent with '
          'the GaussianProcess.')
      
    out = self._mean_func(x,diff,*self._func_args)
    return out

  # convert _cov_func to a class method
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

    if self._dim is not None:
      if x1.shape[1] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *x1* is inconsistent with '
          'the GaussianProcess.')

      if x2.shape[1] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *x2* is inconsistent with '
          'the GaussianProcess.')

      if diff1.shape[0] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *diff1* is inconsistent with '
          'the GaussianProcess.')

      if diff2.shape[0] != self._dim:
        raise ValueError(
          'The number of spatial dimensions for *diff2* is inconsistent with '
          'the GaussianProcess.')

    out = self._cov_func(x1,x2,diff1,diff2,*self._func_args)
    return out
    
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
    This function may raise a warning saying that the covariance 
    matrix is not positive definite. This may be due to numerical 
    rounding error and, if so, the warning can be ignored.  Consider 
    generating a covariance matrix with the *covariance* method and 
    ensuring that its eigenvalues are all positive or effectively zero 
    to within some tolerance.
        
    '''
    mean,cov = self(x,diff=diff)
    return _draw_sample(mean,cov,tol=tol)
    
  def is_positive_definite(self,x,tol=1e-10):
    '''     
    Tests if the covariance matrix, which is the covariance function 
    evaluated at *x*, is positive definite by checking if all the 
    eigenvalues are real and positive. The results of this test do not 
    necessarily indicate whether the covariance function is positive 
    definite.
    
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
    out = _test_positive_definite(cov,tol)
    return out  
    
    
class PriorGaussianProcess(GaussianProcess):
  ''' 
  A *PriorGaussianProcess* instance represents a stationary Gaussian 
  process process which has a constant mean and a covariance function 
  described by a radial basis function, *f*.  Specifically, the 
  Gaussian process, *u*, has a mean and covariance described as
  
    mean(u(x)) = b
    
    cov(u(x),u(x')) = a*f(||x - x'||/c),
    
  where ||*|| denotes the L2 norm, and a, b, and c are user defined 
  parameters. 
  
  The prior can also be given a null space containing all polynomials 
  of order *order*.  If *order* is -1, which is the default, then the 
  prior is given no null space.  If *order* is 0 then the likelihood 
  of a realization of u is invariant to the addition of a constant. If 
  *order* is 1 then the likelihood of a realization of u is invariant 
  to the addition of a constant and linear term, etc.
  
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
  >>> x = np.array([[0.0],[0.5],[1.0],[1.5],[2.0]])
  >>> u = np.array([0.5,1.5,1.25,1.75,1.0])
  >>> sigma = np.array([0.1,0.1,0.1,0.1,0.1])
  >>> gp = gp.posterior(x,u,sigma)
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

