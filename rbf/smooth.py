''' 
provides a function for smoothing large, multidimensional, noisy data sets
'''
import numpy as np
import rbf.fd
import rbf.basis
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from rbf.interpolate import _in_hull

class _IterativeVariance:
  ''' 
  Computes variances of a random process while the samples are being 
  generated. This is more memory efficient than first taking all the 
  samples and then computing the variance.
  '''
  def __init__(self,mean):
    self.mean = mean
    self.sum_squared_diff = np.zeros(mean.shape)
    self.count = 0

  def add_sample(self,sample):
    self.sum_squared_diff += (self.mean - sample)**2
    self.count += 1

  def get_variance(self):
    return self.sum_squared_diff / self.count


def _mask(x,sigma,kind):
  ''' 
  Returns an (N,) boolean array identifying where a smoothed estimate 
  should be made.  
  
    * If kind is 'none', then the smoothed solution will not be 
      estimated at positions where the data uncertainty is inf (i.e. 
      the data is missing).
  
    * If kind is 'interpolate' then missing data will be filled in as 
      long as the position of the missing data is within a convex hull 
      defined by the positions of non-missing data.
  
    * If kind is 'extrapolate' then the smoothed solution will be
      estimated for all positions.
  
  '''
  data_is_missing = np.isinf(sigma) 
  data_is_not_missing = ~data_is_missing
  N = sigma.shape[0]

  if kind == 'none':
    mask = data_is_missing

  elif kind == 'interpolate':
    mask = ~_in_hull(x,x[data_is_not_missing])

  elif kind == 'extrapolate':
    mask = np.zeros(sigma.shape,dtype=bool)

  else:
    raise ValueError('*kind* must be "none", "interpolate", or "extrapolate"')

  return mask


def _average_shortest_distance(x):
  if x.shape[0] == 0:
    return np.inf
  else:
    T = cKDTree(x)
    out = np.mean(T.query(x,2)[0][:,1])
    return out
                                    

def _default_cutoff(x):
  return 1.0/(20*_average_shortest_distance(x))


def _sigma_bar(sigma):
  if sigma.shape[0] == 0:
    return np.inf
  else:  
    return np.sqrt(1.0/np.mean(1.0/sigma**2))
  

def _penalty(cutoff,sigma):
  return (2*np.pi*cutoff)**2*_sigma_bar(sigma)


def smooth(x,u,sigma=None,
           cutoff=None, 
           fill='extrapolate',
           order=2,
           samples=100,
           **kwargs):
  ''' 
  Smooths noisy data in a Bayesian framework by assuming a prior 
  covariance model for the underlying signal.  The prior covariance is 
  chosen such that the power spectral density of the mean of the 
  posterior is effectively zero above a user specified cutoff 
  frequency. This function can be thought of as a low-pass filter with 
  the flexibility to handle observations with variable uncertainties 
  or nonuniformly spaced data in D-dimensional space.
  
  Parameters
  ----------
    x : (N,D) array
      observations points
    
    u : (N,) array, 
      observations at x
    
    sigma : (N,) array, optional
      one standard deviation uncertainty on the observations  
    
    cutoff : float, optional
      cutoff frequency. Frequencies greater than this value will be 
      damped out
      
    order : int, optional
      smoothness order.  Higher orders will cause the frequency 
      response to be more box-like, while lower orders have a 
      frequency response that is tapered across the cutoff frequency.  
      This should almost always be kept at 2 because higher orders 
      tend to be numerically unstable and can produce undesirable 
      ringing artifacts. Also, if D is 2 or greater then the order 
      should be even.
      
    samples : int, optional
      number of posterior samples used to estimate the uncertainty
      
    fill : str, optional
      indicates how to treat missing data (i.e. data with np.inf 
      uncertainty).  Either 'none', 'interpolate', or 'extrapolate'. 
    
  Returns
  -------
    post_mean : (N,) array
    
    post_sigma : (N,) array
  
  '''    
  x = np.asarray(x)
  u = np.asarray(u)  
  u = np.nan_to_num(u)
  u_shape = u.shape
  N,D = x.shape
  if sigma is None:
    sigma = np.ones(u_shape)

  if cutoff is None:
    cutoff = _default_cutoff(x)

  # throw out points where we do not want to estimate the solution
  keep_idx, = np.nonzero(~_mask(x,sigma,fill))
  K = len(keep_idx)
  x = x[keep_idx]
  u = u[keep_idx]
  sigma = sigma[keep_idx]
  # build data weight matrix, which is the inverse of the Cholesky 
  # decomposition of the data covariance
  Wdata = 1.0/sigma
  Wrow,Wcol = range(K),range(K)
  W = scipy.sparse.csr_matrix((Wdata,(Wrow,Wcol)),(K,K))
  # build differentiation matrix, which is the inverse of the Choleksy 
  # decomposition of the prior covariance
  diff = order*np.eye(D,dtype=int)
  if D == 1:
    # if one dimensional, then use adjacency rather than nearest 
    # neighbors to form stencils
    L = rbf.fd.diff_matrix_1d(x,diff,**kwargs)
  else:
    L = rbf.fd.diff_matrix(x,diff,**kwargs)
    
  # penalty used to scale the prior
  p = _penalty(cutoff,sigma)
  L *= 1.0/p
  # form left and right hand side of the system to solve
  lhs = W.T.dot(W) + L.T.dot(L)
  rhs = W.T.dot(W).dot(u)
  # generate LU decomposition of left-hand side
  lu = spla.splu(lhs)
  # compute mean of posterior
  post_mean = lu.solve(rhs)
  # compute the posterior standard deviation
  ivar = _IterativeVariance(post_mean)
  for i in xrange(samples):
    w1 = np.random.normal(0.0,1.0,K)
    w2 = np.random.normal(0.0,1.0,K)
    # generate sample of the posterior
    post_sample = lu.solve(rhs + W.T.dot(w1) + L.T.dot(w2))
    ivar.add_sample(post_sample)
    
  post_sigma = np.sqrt(ivar.get_variance())
  # expand the mean and standard deviation to the original size. 
  # points which were not smoothed are returned with mean=np.nan and 
  # sigma=np.inf
  post_mean_full = np.empty(N)
  post_mean_full[:] = np.nan
  post_mean_full[keep_idx] = post_mean
  post_sigma_full = np.empty(N)
  post_sigma_full[:] = np.inf
  post_sigma_full[keep_idx] = post_sigma

  return post_mean_full,post_sigma_full
  

          
