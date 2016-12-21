import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
import rbf.fd
import rbf.poly
import rbf.basis
from sympy import exp
np.random.seed(1)

#####################################################################
# Define Gaussian processs base class
#####################################################################    
class GaussianProcess(object):
  def __init__(self,mean_func,cov_func,*args,**kwargs):
    ''' 
    Base class for Gaussian Processes. This is not intended for the 
    end-user.
    
    Parameters
    ----------
    mean_func : function
      Function whose positional arguments are an array of positions 
      followed by *args*. This function must also have a *diff* key 
      word argument.

    cov_func : function
      Function whose positional arguments are two arrays of positions 
      followed by *args*. This function must have a *diff1* and a 
      *diff2* key word argument.
    
    *args : tuple
      Additional positional arguments passed to *mean_func* and 
      *cov_func*      

    *kwargs : tuple
      Additional key word arguments passed to *mean_func* and 
      *cov_func*

    '''
    self._mean_func = mean_func
    self._cov_func = cov_func
    self._args = args
    self._kwargs = kwargs
  
  def __call__(self,x,diff=None):
    mean = self.mean(x,diff=diff)
    cov = self.covariance(x,x,diff1=diff,diff2=diff)
    return mean,cov

  def __add__(self,other):
    return self.sum(other)

  def __sub__(self,other):
    return self.difference(other)

  def __mul__(self,c):
    return self.scale(c)

  def __rmul__(self,c):
    return self.__mul__(c)

  def sum(self,other):
    def mean_func(x,diff,gp1,gp2):
      out = gp1.mean(x,diff=diff) + gp2.mean(x,diff=diff)
      return out       

    def cov_func(x1,x2,diff1,diff2,gp1,gp2):
      out = (gp1.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             gp2.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out
            
    out = GaussianProcess(mean_func,cov_func,self,other)
    return out

  def difference(self,other):
    def mean_func(x,diff,gp1,gp2):
      out = gp1.mean(x,diff=diff) - gp2.mean(x,diff=diff)
      return out
      
    def cov_func(x1,x2,diff1,diff2,gp1,gp2):
      out = (gp1.covariance(x1,x2,diff1=diff1,diff2=diff2) + 
             gp2.covariance(x1,x2,diff1=diff1,diff2=diff2))
      return out       
            
    out = GaussianProcess(mean_func,cov_func,self,other)
    return out
    
  def scale(self,c):
    def mean_func(x,diff,c,gp):
      out = c*gp.mean(x,diff=diff)
      return out

    def cov_func(x1,x2,diff1,diff2,c,gp):
      out = c**2*gp.covariance(x1,x2,diff1=diff1,diff2=diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,c,self)
    return out

  def derivative(self,d):
    ''' 
    returns a derivative of the Gaussian process
    '''
    d = np.asarray(d,dtype=int)
    def mean_func(x,diff,d,gp):
      if (d.shape[0] != x.shape[1]) | (diff.shape[0] != x.shape[1]): 
        raise ValueError(
          'length of derivative specifications must be equal to the '
          'spatial dimensions of x')
        
      out = gp.mean(x,diff=d+diff)
      return out 

    def cov_func(x1,x2,diff1,diff2,d,gp):
      if ((d.shape[0] != x1.shape[1]) | 
          (diff1.shape[0] != x1.shape[1]) | 
          (diff2.shape[0] != x1.shape[1])): 
        raise ValueError(
          'length of derivative specifications must be equal to the '
          'spatial dimensions of x')

      out = gp.covariance(x1,x2,diff1=d+diff1,diff2=d+diff2)
      return out
      
    out = GaussianProcess(mean_func,cov_func,d,self)
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

    def mean_func(x,diff,obs_x,obs_mu,obs_sigma,obs_diff,gp):
      res = obs_mu - gp.mean(obs_x,diff=obs_diff)
      Cd = np.diag(obs_sigma**2)
      K = gp.covariance(obs_x,obs_x,diff1=obs_diff,diff2=obs_diff) + Cd
      Kinv = np.linalg.inv(K)
      Ki = gp.covariance(x,obs_x,diff1=diff,diff2=obs_diff)
      out = gp.mean(x,diff=diff) + Ki.dot(Kinv.dot(res))
      return out

    def cov_func(x1,x2,diff1,diff2,obs_x,obs_mu,obs_sigma,obs_diff,gp):
      Cd = np.diag(obs_sigma**2)
      K = gp.covariance(obs_x,obs_x,diff1=obs_diff,diff2=obs_diff) + Cd
      Kinv = np.linalg.inv(K)
      Ki = gp.covariance(x1,obs_x,diff1=diff1,diff2=obs_diff)
      Kj = gp.covariance(obs_x,x2,diff1=obs_diff,diff2=diff2)
      Kij = gp.covariance(x1,x2,diff1=diff1,diff2=diff2)
      out = Kij - Ki.dot(Kinv).dot(Kj)        
      return out

    out = GaussianProcess(mean_func,cov_func,obs_x,obs_mu,obs_sigma,obs_diff,self)
    return out

  # convert _mean_func to a class method
  def mean(self,x,diff=None):
    x = np.asarray(x)
    if diff is None:  
      diff = np.zeros(x.shape[1],dtype=int)
    else:
      diff = np.asarray(diff,dtype=int)

    out = self._mean_func(x,diff,*self._args,**self._kwargs)
    return out

  # convert _cov_func to a class method
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

    out = self._cov_func(x1,x2,diff1,diff2,*self._args,**self._kwargs)
    return out
    
  def draw_sample(self,x,diff=None):  
    mean,cov = self(x,diff=diff)
    return np.random.multivariate_normal(mean,cov)
    

#####################################################################
# define mean and covariance function for squared exponential GP
#####################################################################
_R = rbf.basis.get_R()
_EPS = rbf.basis.get_EPS()
_se = rbf.basis.RBF(exp(-_R**2/(2*_EPS**2)))

def _se_mean_func(x,diff,mu,sigma,cls):
  if sum(diff) == 0:
    return mu*np.ones(x.shape[0])       
  else:
    return np.zeros(x.shape[0])  

def _se_cov_func(x1,x2,diff1,diff2,mu,sigma,cls):
  eps = cls*np.ones(x2.shape[0])
  coeff = (-1)**sum(diff2)*sigma**2
  diff = diff1 + diff2
  K = coeff*_se(x1,x2,eps=eps,diff=diff)
  return K

def squared_exp_prior(mu,sigma,cls):
  out = GaussianProcess(_se_mean_func,_se_cov_func,mu,sigma,cls)
  return out


if __name__ == '__main__':
  N = 300
  Nitp = 1000
  x = np.linspace(0.0,10.0,N)[:,None]
  xitp = np.linspace(0.0,10.0,Nitp)[:,None]
  prior_mu = 0.0
  prior_sigma = 1.0
  prior_cls = 0.5

  obs_sigma = 0.1*np.ones(N)
  obs_mu = np.cos(x[:,0]) + np.random.normal(0.0,obs_sigma)
  obs_mu2 = np.sin(x[:,0]) + np.random.normal(0.0,obs_sigma[:])

  gp = squared_exp_prior(prior_mu,prior_sigma,prior_cls)
  gp = gp.posterior(x,obs_mu,obs_sigma,obs_diff=(1,))
  gp = gp.posterior(x[::5],obs_mu2[::5],obs_sigma[::5],obs_diff=(0,))
  mean,cov = gp(xitp,diff=(0,))
  std = np.sqrt(np.diag(cov))

  plt.errorbar(x,obs_mu,obs_sigma,fmt='ko')
  plt.errorbar(x[::5],obs_mu2[::5],obs_sigma[::5],fmt='go')
  plt.plot(xitp,mean,'r-')
  plt.plot(xitp,np.sin(xitp[:,0]),'b-')
  plt.fill_between(xitp[:,0],mean-std,mean+std,color='r',alpha=0.2)
  plt.show()
  quit()

