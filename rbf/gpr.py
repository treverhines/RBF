''' 
Module for Gaussian process regression.
'''
import numpy as np
import rbf.fd
import rbf.poly
import rbf.basis
from functools import wraps

def _block_if_null_space_exists(fin):
  @wraps(fin)
  def fout(self,*args,**kwargs):
    if self._order != -1: 
      raise ValueError(
        'The method *%s* cannot be used for a GaussianProcess with a '
        'null space. Use the *posterior* method to construct a new '
        'GaussianProcess where the null space has been removed by '
        'data constraints.' 
        % fin.__name__)
    else:
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
  
  @_block_if_null_space_exists
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

  @_block_if_null_space_exists
  def __add__(self,other):
    ''' 
    gp1 + gp2 <==> gp1.sum(gp2)
    '''
    return self.sum(other)

  @_block_if_null_space_exists
  def __sub__(self,other):
    ''' 
    gp1 - gp2 <==> gp1.difference(gp2)
    '''
    return self.difference(other)

  @_block_if_null_space_exists
  def __mul__(self,c):
    ''' 
    c*gp  <==> gp.scale(c)
    '''
    return self.scale(c)

  @_block_if_null_space_exists
  def __rmul__(self,c):
    ''' 
    gp*c  <==> gp.scale(c)
    '''
    return self.__mul__(c)

  @_block_if_null_space_exists
  def sum(self,other):
    ''' 
    Adds two Gaussian processes
    
    Parameters
    ----------
    other : GuassianProcess 
      
    Returns
    -------
    out : GaussianProcess 
      
    '''
    def mean_func(x,diff,gp1,gp2):
      out = gp1.mean(x,diff=diff) + gp2.mean(x,diff=diff)
      return out       

    def cov_func(x1,x2,diff1,diff2,gp1,gp2):
      out = (gp1.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             gp2.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out
            
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(self,other))
    return out

  @_block_if_null_space_exists
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
    def mean_func(x,diff,gp1,gp2):
      out = gp1.mean(x,diff=diff) - gp2.mean(x,diff=diff)
      return out
      
    def cov_func(x1,x2,diff1,diff2,gp1,gp2):
      out = (gp1.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             gp2.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out       
            
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(self,other))
    return out
    
  @_block_if_null_space_exists
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
    def mean_func(x,diff,c_,gp):
      out = c_*gp.mean(x,diff=diff)
      return out

    def cov_func(x1,x2,diff1,diff2,c_,gp):
      out = c_**2*gp.covariance(x1,x2,diff1=diff1,diff2=diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(c,self))
    return out

  @_block_if_null_space_exists
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
    out_diff = np.asarray(diff,dtype=int)
    def mean_func(x,diff,out_diff_,gp):
      out = gp.mean(x,diff=out_diff_+diff)
      return out 

    def cov_func(x1,x2,diff1,diff2,out_diff_,gp):
      out = gp.covariance(x1,x2,
                          diff1=out_diff_+diff1,
                          diff2=out_diff_+diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(out_diff,self),
                          dim=out_diff.shape[0])
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
    obs_x = np.asarray(x)
    obs_mu = np.asarray(mu)
    if diff is None:
      obs_diff = np.zeros(obs_x.shape[1],dtype=int)
    else:
      obs_diff = np.asarray(diff,dtype=int)
    
    if sigma is None:
      obs_sigma = np.zeros(obs_x.shape[0])      
    else:
      obs_sigma = np.asarray(sigma)

    def mean_func(x,diff,obs_x_,obs_mu_,obs_sigma_,
                  obs_diff_,gp):
      # the mean and covariance of the prior cannot be evaluated 
      # unless the null space order is set to -1.  
      order = gp._order
      gp._order = -1
      try:
        N,D = obs_x_.shape
        I = x.shape[0]
        # determine the powers for each monomial spanning the null 
        # space
        powers = rbf.poly.powers(order,D) 
        P = powers.shape[0]
        # difference between the prior mean and the observations
        res = np.zeros(N+P)
        res[:N] = obs_mu_ - gp.mean(obs_x_,diff=obs_diff_)
        # data covariance matrix
        Cd = np.diag(obs_sigma_**2)
        # prior covariance between observations
        K = gp.covariance(obs_x_,obs_x_,diff1=obs_diff_,diff2=obs_diff_)
        # polynomial matrix evaluated at the observation points 
        H = rbf.poly.mvmonos(obs_x_,powers,diff=obs_diff_)
        # combine the covariances and the polynomial matrices 
        A = np.zeros((N+P,N+P))
        A[:N,:N] = K + Cd
        A[:N,N:] = H
        A[N:,:N] = H.T
        Ainv = np.linalg.inv(A)
        # covariance between the interpolation points and the 
        # observation points
        Ki = gp.covariance(x,obs_x_,diff1=diff,diff2=obs_diff_)
        # polynomial evaluated at the interpolation points
        Hi = rbf.poly.mvmonos(x,powers,diff=diff)
        # form the interpolation matrix
        Ai = np.zeros((I,N+P))
        Ai[:,:N] = Ki
        Ai[:,N:] = Hi
      
        out = gp.mean(x,diff=diff) + Ai.dot(Ainv.dot(res))
      
      except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
          'Failed to compute the posterior mean. This is likely '
          'because there is not enough data to constrain a null '
          'space in the prior')
          
      finally:
        # make sure the null space order is returned to the original 
        # value
        gp._order = order

      return out

    def cov_func(x1,x2,diff1,diff2,obs_x_,obs_mu_,obs_sigma_,
                 obs_diff_,gp):
      # the mean and covariance of the prior cannot be evaluated 
      # unless the null space order is set to -1.  
      order = gp._order
      gp._order = -1
      try:
        N,D = obs_x_.shape
        # I and J will be the dimensions of the output covariance 
        # matrix
        I = x1.shape[0]
        J = x2.shape[0]
        # powers for each monomial spanning the null space
        powers = rbf.poly.powers(order,D) 
        P = powers.shape[0]
        # data covariance matrix      
        Cd = np.diag(obs_sigma_**2)
        # prior covariance between observation points
        K = gp.covariance(obs_x_,obs_x_,diff1=obs_diff_,diff2=obs_diff_)
        # polynomial matrix evaluated at the observation points   
        H = rbf.poly.mvmonos(obs_x_,powers,diff=obs_diff_)
        # combine the covariance and the polynomial matrices
        A = np.zeros((N+P,N+P))
        A[:N,:N] = K + Cd
        A[:N,N:] = H
        A[N:,:N] = H.T
        Ainv = np.linalg.inv(A)
        # covariance between the interpolation points      
        Kii = gp.covariance(x1,x2,diff1=diff1,diff2=diff2)
        # covariance between x1 and the observation points
        Ki = gp.covariance(x1,obs_x_,diff1=diff1,diff2=obs_diff_)
        # polynomial matrix evaluated at x1
        Hi = rbf.poly.mvmonos(x1,powers,diff=diff1)
        # interpolation matrix for x1
        Ai = np.zeros((I,N+P))
        Ai[:,:N] = Ki
        Ai[:,N:] = Hi
        # covariance between x2 and the observation points
        Kj = gp.covariance(x2,obs_x_,diff1=diff2,diff2=obs_diff_)
        # polynomial matrix evaluated at x2
        Hj = rbf.poly.mvmonos(x2,powers,diff=diff2)
        # interpolation matrix for x2
        Aj = np.zeros((J,N+P))
        Aj[:,:N] = Kj
        Aj[:,N:] = Hj

        out = Kii - Ai.dot(Ainv).dot(Aj.T) 

      except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
          'Failed to compute the posterior covariance. This is '
          'likely because there is not enough data to constrain a '
          'null space in the prior.')

      finally:
        # make sure the order is reset
        gp._order = order
        
      return out

    out = GaussianProcess(mean_func,cov_func,
                          func_args=(obs_x,obs_mu,obs_sigma,obs_diff,self),
                          dim=obs_x.shape[1])
    return out

  # convert _mean_func to a class method
  @_block_if_null_space_exists
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
  @_block_if_null_space_exists
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
      diff2 = np.zeros(x1.shape[1],dtype=int)
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
    
  @_block_if_null_space_exists
  def draw_sample(self,x,diff=None):  
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
    return np.random.multivariate_normal(mean,cov)
    
    
class PriorGaussianProcess(GaussianProcess):
  ''' 
  A *PriorGaussianProcess* instance represents a stationary Gaussian 
  process process which has a constant mean and a covariance function 
  described by a radial basis function, *f*.  Specifically, the 
  Gaussian process, *u*, has a mean and covariance described as
  
    mean(u(x)) = b
    
    cov(u(x1),u(x2)) = a*f(||x1 - x2||/c),
    
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
  
  Not all radial basis functions are positive definite.  Care must be 
  taken to ensure that the choice of *basis* and *order* are 
  meaningful. 

  '''
  def __init__(self,basis,coeff,order=-1):
    def mean_func(x,diff,basis_,coeff_):
      if sum(diff) == 0:
        out = coeff_[1]*np.ones(x.shape[0])
      else:  
        out = np.zeros(x.shape[0])

      return out
      
    def cov_func(x1,x2,diff1,diff2,basis_,coeff_):
      eps = np.ones(x2.shape[0])/coeff_[2]
      a = (-1)**sum(diff2)*coeff_[0]
      diff = diff1 + diff2
      out = a*basis_(x1,x2,eps=eps,diff=diff)
      if np.any(~np.isfinite(out)):
        raise ValueError(
          'Encountered a non-finite covariance. This is likely '
          'because the prior basis function is not sufficiently '
          'differentiable.')

      return out
      
    GaussianProcess.__init__(self,mean_func,cov_func,
                             func_args=(basis,coeff),
                             order=order)

