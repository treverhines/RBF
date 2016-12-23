''' 
Module for Gaussian process regression.
'''

import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import rbf.poly
import rbf.basis
from functools import wraps

def _block_if_null_space_exists(fin):
  @wraps
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
  def __init__(self,mean_func,cov_func,func_args=(),order=-1,dim=None):
    ''' 
    A GaussianProcess instance represents a stochastic process which 
    is defined in terms of its mean and covariance function.  This 
    class allows for basic operations on Gaussian processes which 
    includes addition, subtraction, scaling, differentiation, 
    sampling, and conditioning.
    
    A Gaussian process should be instantiated with one of the prior 
    function, (e.g. squared_exp_prior).
    
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
    self._mean_func = mean_func
    self._cov_func = cov_func
    self._func_args = func_args
    self._order = order
    self._dim = dim
  
  @_block_if_null_space_exists
  def __call__(self,x,diff=None):
    mean = self.mean(x,diff=diff)
    cov = self.covariance(x,x,diff1=diff,diff2=diff)
    return mean,cov

  @_block_if_null_space_exists
  def __add__(self,other):
    return self.sum(other)

  @_block_if_null_space_exists
  def __sub__(self,other):
    return self.difference(other)

  @_block_if_null_space_exists
  def __mul__(self,c):
    return self.scale(c)

  @_block_if_null_space_exists
  def __rmul__(self,c):
    return self.__mul__(c)

  @_block_if_null_space_exists
  def sum(self,other):
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
    def mean_func(x,diff,c,gp):
      out = c*gp.mean(x,diff=diff)
      return out

    def cov_func(x1,x2,diff1,diff2,c,gp):
      out = c**2*gp.covariance(x1,x2,diff1=diff1,diff2=diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(c,self))
    return out

  @_block_if_null_space_exists
  def derivative(self,d):
    ''' 
    returns a derivative of the Gaussian process
    '''
    d = np.asarray(d,dtype=int)
    def mean_func(x,diff,d,gp):
      out = gp.mean(x,diff=d+diff)
      return out 

    def cov_func(x1,x2,diff1,diff2,d,gp):
      out = gp.covariance(x1,x2,diff1=d+diff1,diff2=d+diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,
                          func_args=(d,self),
                          dim=d.shape[0])
    return out
  
  def posterior(self,obs_x,obs_mu,obs_sigma=None,obs_diff=None):
    obs_x = np.asarray(obs_x)
    obs_mu = np.asarray(obs_mu)
    if obs_diff is None:
      obs_diff = np.zeros(obs_x.shape[1],dtype=int)
      
    if obs_sigma is None:
      obs_sigma = np.zeros(obs_x.shape[0])      
    else:
      obs_sigma = np.asarray(obs_sigma)

    def mean_func(x,diff,obs_x,obs_mu,obs_sigma,
                  obs_diff,gp):
      order = gp._order
      gp._order = -1
      try:
        N,D = obs_x.shape
        I = x.shape[0]
        powers = rbf.poly.powers(order,D) 
        P = powers.shape[0]

        res = np.zeros(N+P)
        res[:N] = obs_mu - gp.mean(obs_x,diff=obs_diff)
      
        A = np.zeros((N+P,N+P))
        Cd = np.diag(obs_sigma**2)
        K = gp.covariance(obs_x,obs_x,diff1=obs_diff,diff2=obs_diff)
        H = rbf.poly.mvmonos(obs_x,powers,diff=obs_diff)
        A[:N,:N] = K + Cd
        A[:N,N:] = H
        A[N:,:N] = H.T
        Ainv = np.linalg.inv(A)
      
        Ki = gp.covariance(x,obs_x,diff1=diff,diff2=obs_diff)
        Hi = rbf.poly.mvmonos(x,powers,diff=diff)
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
        gp._order = order

      return out

    def cov_func(x1,x2,diff1,diff2,obs_x,obs_mu,obs_sigma,
                 obs_diff,gp):
      order = gp._order
      gp._order = -1
      try:
        N,D = obs_x.shape
        I = x1.shape[0]
        J = x2.shape[0]
        powers = rbf.poly.powers(order,D) 
        P = powers.shape[0]
      
        Cd = np.diag(obs_sigma**2)
        K = gp.covariance(obs_x,obs_x,diff1=obs_diff,diff2=obs_diff)
        H = rbf.poly.mvmonos(obs_x,powers,diff=obs_diff)
        A = np.zeros((N+P,N+P))
        A[:N,:N] = K + Cd
        A[:N,N:] = H
        A[N:,:N] = H.T
        Ainv = np.linalg.inv(A)
      
        Kii = gp.covariance(x1,x2,diff1=diff1,diff2=diff2)

        Ki = gp.covariance(x1,obs_x,diff1=diff1,diff2=obs_diff)
        Hi = rbf.poly.mvmonos(x1,powers,diff=diff1)
        Ai = np.zeros((I,N+P))
        Ai[:,:N] = Ki
        Ai[:,N:] = Hi

        Kj = gp.covariance(x2,obs_x,diff1=diff2,diff2=obs_diff)
        Hj = rbf.poly.mvmonos(x2,powers,diff=diff2)
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
        gp._order = order
        
      return out

    out = GaussianProcess(mean_func,cov_func,
                          func_args=(obs_x,obs_mu,obs_sigma,obs_diff,self),
                          dim=obs_x.shape[1])
    return out

  # convert _mean_func to a class method
  @_block_if_null_space_exists
  def mean(self,x,diff=None):
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
    mean,cov = self(x,diff=diff)
    return np.random.multivariate_normal(mean,cov)
    
    
#####################################################################
# Define prior models
#####################################################################

def _squared_exp_mean_func(x,diff,mu,sigma,cls):
  if sum(diff) == 0:
    return mu*np.ones(x.shape[0])       
  else:
    return np.zeros(x.shape[0])  

def _squared_exp_cov_func(x1,x2,diff1,diff2,mu,sigma,cls):
  eps = np.ones(x2.shape[0])/(np.sqrt(2)*cls)
  coeff = (-1)**sum(diff2)*sigma**2
  diff = diff1 + diff2
  K = coeff*rbf.basis.ga(x1,x2,eps=eps,diff=diff)
  return K

def squared_exp_prior(mu,sigma,cls):
  out = GaussianProcess(_squared_exp_mean_func,
                        _squared_exp_cov_func,
                        func_args=(mu,sigma,cls))
  return out


def _spline1d_mean_func(x,diff,p):
  return np.zeros(x.shape[0])  

def _spline1d_cov_func(x1,x2,diff1,diff2,p):
  eps = np.ones(x2.shape[0])/p
  coeff = (-1)**sum(diff2)
  diff = diff1 + diff2
  if sum(diff) > 3:
    raise ValueError(
      'The prior model is not sufficiently differentiable')
  
  K = coeff*rbf.basis.phs3(x1,x2,eps=eps,diff=diff)
  return K

def spline1d_prior(p):
  out = GaussianProcess(_spline1d_mean_func,
                        _spline1d_cov_func,
                        func_args=(p,),
                        dim=1,order=1)
  return out


def _spline2d_mean_func(x,diff,p):
  return np.zeros(x.shape[0])  

def _spline2d_cov_func(x1,x2,diff1,diff2,p):
  eps = np.ones(x2.shape[0])/p
  coeff = (-1)**sum(diff2)
  diff = diff1 + diff2
  K = coeff*rbf.basis.phs2(x1,x2,eps=eps,diff=diff)
  return K

def spline2d_prior(p):
  out = GaussianProcess(_spline2d_mean_func,
                        _spline2d_cov_func,
                        func_args=(p,),
                        dim=2,order=1)
  return out
  
  

if __name__ == '__main__':
  N = 10
  Nitp = 1000
  #x = np.linspace(0.0,1.0,N)[:,None]
  x = np.sort(np.random.uniform(0.0,1.0,N))[:,None]
  xitp = np.linspace(0.0,1.0,Nitp)[:,None]
  prior_mu = 0.0
  prior_sigma = 0.1
  prior_cls = 0.1

  obs_sigma = 0.1*np.ones(N)
  obs_mu = 5*np.sin(x[:,0]) + np.random.normal(0.0,obs_sigma)

  gp = spline1d_prior(0.4)
  gp = gp.posterior(x[:5],obs_mu[:5],obs_sigma[:5],obs_diff=(1,))
  gp = gp.posterior(x[5:],obs_mu[5:],obs_sigma[5:])
  gp = gp.derivative((1,))
  #print(gp.order)
  #gp = gp2
  
  mean,cov = gp(xitp)
  val,vec = np.linalg.eig(cov)
  print(np.min(val))
  std = np.sqrt(np.diag(cov))

  plt.errorbar(x,obs_mu,obs_sigma,fmt='ko')
  plt.plot(xitp,mean,'r-')
  for i in range(3): plt.plot(xitp,gp.draw_sample(xitp),'r--')
  plt.fill_between(xitp[:,0],mean-std,mean+std,color='r',alpha=0.2)
  plt.show()
  quit()

