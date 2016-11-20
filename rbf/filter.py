''' 
This module provides a function for denoising large, scattered,
multidimensional data sets. The filtered solution returned by this
function is constrained to resemble the observed data while also
minimizing its Laplacian. The Laplacian operator is computed with the
RBF-FD method and it is sparse, making it suitable for large scale
problems.

'''
import numpy as np
import rbf.fd
import rbf.mp
import rbf.basis
from rbf.poly import memoize
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from rbf.interpolate import _in_hull
import logging
logger = logging.getLogger(__name__)

class _IterativeVariance:
  ''' 
  Computes variances of a random process while the samples are being 
  generated. This is more memory efficient than first taking all the 
  samples and then computing the variance.
  '''
  def __init__(self,mean):
    mean = np.asarray(mean)
    self.mean = mean
    self.sum_squared_diff = np.zeros(mean.shape)
    self.count = 0

  def add_sample(self,sample):
    sample = np.asarray(sample)
    self.sum_squared_diff += (self.mean - sample)**2
    self.count += 1

  def get_variance(self):
    return self.sum_squared_diff / self.count


def _get_mask(x,sigma,kind):
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
  

  Parameters
  ----------
  x: (N,D) array
    observation points
    
  sigma: (N,) array
    uncertainties for each observation point. np.inf indicates that 
    the observation is missing
      
  kind: str
    either 'none', 'interpolate', or 'extrapolate'  
      
  '''
  data_is_missing = np.isinf(sigma) 
  if kind == 'none':
    mask = data_is_missing

  elif kind == 'interpolate':
    mask = ~_in_hull(x,x[~data_is_missing])

  elif kind == 'extrapolate':
    mask = np.zeros(sigma.shape,dtype=bool)

  else:
    raise ValueError('*kind* must be "none", "interpolate", or "extrapolate"')

  return mask


def _average_shortest_distance(x):
  ''' 
  returns the average shortest distance between points in x. 
  '''
  if x.shape[0] == 0:
    return np.inf
  else:
    T = cKDTree(x)
    out = np.mean(T.query(x,2)[0][:,1])
    return out
                                    

def _default_cutoff(x):
  '''  
  the default cutoff frequency has a corresponding wavelength that is 
  20 times the average shortest distance between observations
  '''
  return 1.0/(20*_average_shortest_distance(x))


def _sigma_bar(sigma):
  ''' 
  returns the characteristic uncertainty
  '''
  if sigma.shape[0] == 0:
    return np.inf
  else:  
    return np.sqrt(1.0/np.mean(1.0/sigma**2))
  

def _penalty(cutoff,order,sigma):
  return (2*np.pi*cutoff)**order*_sigma_bar(sigma)


def _diag(diag):
  ''' 
  returns a diagonal csr matrix. Unlike scipy.sparse.diags, this 
  properly handles zero-length input
  '''
  K = len(diag)
  r,c = range(K),range(K)
  out = scipy.sparse.csr_matrix((diag,(r,c)),(K,K))
  return out
     

def filter(x,u,sigma=None,
           cutoff=None, 
           fill='extrapolate',
           order=2,
           samples=100,
           diffs=None,
           procs=0,
           **kwargs):
  ''' 
  Effectively performs a low-pass filter over a noisy, scattered, 
  multidimensional data set. This function makes use of sparse RBF-FD 
  differentiation matrices and is ideal for smoothing large data sets. 
  The details on the theory for this function will be in an upcoming 
  manuscript.
  
  Parameters
  ----------
  x : (N,D) array
    Observation locations
    
  u : (..., N) array, 
    data values at x
    
  sigma : (..., N) array, optional
    One standard deviation uncertainty on the observations. This 
    must have the same shape as u. Any np.inf entries are treated as 
    masked data.  Masked data can either be ignored or filled in 
    depending on the *fill* argument. If *sigma* is not provided 
    then it defaults to an array of ones
    
  cutoff : float, optional
    Cutoff frequency. Frequencies greater than this value will be 
    damped out. This defaults to a frequency corresponding to a 
    wavelength which is 20 times the average shortest distance 
    between points in *x*
      
  order : int, optional
    Smoothness order.  Higher orders will cause the frequency 
    response to be more box-like, while lower orders have a 
    frequency response that is tapered across the cutoff frequency.  
    This should almost always be kept at 2 because higher orders 
    tend to be numerically unstable and can produce undesirable 
    ringing artifacts. Also, if D is 2 or greater then the order 
    should be even
      
  samples : int, optional
    The uncertainty on the filtered solution is estimated by finding 
    the standard deviation of random perturbations to the data vector. 
    This argument specifies the number of random perturbations to use. 
    Increasing this value will increase the accuracy of the 
    uncertainty estimate as well as the computation time
      
  fill : str, optional
    Indicates how to treat missing data (i.e. data where *sigma* is 
    np.inf).  Either 'none', 'interpolate', or 'extrapolate'. If 
    'none' then missing data is ignored and the returned mean and 
    uncertainty at those observation points will be np.nan and 
    np.inf respectively. If *fill* is 'interpolate' then a smoothed 
    solution will be estimated at missing interior observation 
    points (i.e. no extrapolation).  If fill is 'extrapolate' then a 
    smoothed solution is estimated at every observation point

  diffs : (D,) or (K,D) int array, optional
    If provided then the output will be a derivative of the smoothed 
    solution. The derivative can be specified with a (D,) array 
    where each entry indicates the derivative order for the 
    corresponding dimension. For example, if the observations exist 
    in two-dimensional space, then the second x derivative can be 
    returned by setting *diffs* to [2,0]. A differential operator 
    can be specified with a (K,D) array. For example the Laplacian 
    can be returned with [[2,0],[0,2]]
      
  procs : int, optional
    Distribute the tasks among this many subprocesses. This defaults 
    to 0 (i.e. the parent process does all the work).  Each task is 
    to evaluate the filtered solution for one of the (N,) arrays in 
    *u* and *sigma*. So if *u* and *sigma* are (N,) arrays then 
    using multiple process will not provide any speed improvement
   
  **kwargs : optional
    Additional key word arguments used to construct the RBF-FD 
    differentiation matrices. There are two differentiation matrices 
    used in this function: one used to construct the prior for the 
    underlying signal, and another to differentiate the filtered 
    solution.  These key word arguments are passed to 
    `rbf.fd.weight_matrix`

  Returns
  -------
  post_mean : (..., N) array
    
  post_sigma : (..., N) array
  
  '''    
  x = np.asarray(x)
  u = np.asarray(u)  
  u = np.nan_to_num(u)
  N,dim = x.shape
  P = int(np.prod(u.shape[:-1])) 
  if sigma is None:
    sigma = np.ones(u.shape)

  if cutoff is None:
    cutoff = _default_cutoff(x)
    
  if diffs is None:
    diffs = np.zeros(dim,dtype=int)  

  # flatten u and sigma to a 2D array
  input_u_shape = u.shape
  u = u.reshape((P,N))
  sigma = sigma.reshape((P,N))
    
  # memoized function to form the differentiation matrices used for 
  # the prior and post-processing
  @memoize
  def build_L_and_D(mask):
    # This function build the differentiation matrix used for the 
    # prior, L, and the differentiation matrix used for 
    # postprocessing, D. Note1: this function makes use of variables 
    # which are outside of its scope. Note2: this function is memoized 
    # and so it may take up a lot of memory if it is called many times 
    # with different masks
    mask = np.asarray(mask,dtype=bool)        
    prior_diffs = order*np.eye(dim,dtype=int)
    L = rbf.fd.weight_matrix(x[~mask],x[~mask],prior_diffs,**kwargs)
    D = rbf.fd.weight_matrix(x[~mask],x[~mask],diffs,**kwargs)
    return L,D  
                
  def calculate_posterior(i):
    # This function calculates the posterior for u[i,:] and 
    # sigma[i,:]. Note: this function makes use of variables which are 
    # outside of its scope.
    logger.debug('evaluating the filtered solution for data set %s ...' % i)

    # identify observation points where we do not want to estimate the 
    # filtered solution
    mask = _get_mask(x,sigma[i],fill)
    # number of unmasked entries
    K = np.sum(~mask)
    # build differentiation matrices
    L,D = build_L_and_D(tuple(mask))
    # form weight matrix
    W = _diag(1.0/sigma[i,~mask])
    # compute penalty parameter
    p = _penalty(cutoff,order,sigma[i,~mask])
    # form left and right hand side of the system to solve
    lhs = W.T.dot(W) + L.T.dot(L)/p**2
    rhs = W.T.dot(W).dot(u[i,~mask])
    # generate LU decomposition of left-hand side
    lu = spla.splu(lhs)
    # compute the smoothed derivative of the posterior mean
    post_mean = np.empty((N,))
    post_mean[~mask] = D.dot(lu.solve(rhs))
    post_mean[mask] = np.nan
    # compute the posterior standard deviation. This is done through 
    # repeated random perturbations of the data and prior vector. 
    ivar = _IterativeVariance(post_mean[~mask])
    for j in range(samples):
      w1 = np.random.normal(0.0,1.0,K)
      w2 = np.random.normal(0.0,1.0,K)
      # generate sample of the posterior
      post_sample = lu.solve(rhs + W.T.dot(w1) + L.T.dot(w2)/p)
      # differentiate the sample
      post_sample = D.dot(post_sample)
      ivar.add_sample(post_sample)
    
    post_sigma = np.empty((N,))
    post_sigma[~mask] = np.sqrt(ivar.get_variance())
    post_sigma[mask] = np.inf
    logger.debug('done')
    return post_mean,post_sigma
    
  # Calculate the posterior for each (N,) array in u and sigma. 
  # This is done in parallel, where *procs* is the number of 
  # subprocesses spawned.
  post = rbf.mp.parmap(calculate_posterior,range(P),workers=procs)
  post_mean = np.array([k[0] for k in post])
  post_sigma = np.array([k[1] for k in post])

  post_mean = post_mean.reshape(input_u_shape)
  post_sigma = post_sigma.reshape(input_u_shape)
  return post_mean,post_sigma  


def filter_eig(x,u,sigma=None,
               cutoff=None, 
               order=2,
               **kwargs):
  ''' 
  Returns the eigenvectors and eigenvalues for the RBF-FD filter
  
  '''                 
  x = np.asarray(x)
  u = np.asarray(u)  
  N,dim = x.shape
  if sigma is None:
    sigma = np.ones(u.shape)

  if cutoff is None:
    cutoff = _default_cutoff(x)
    
  prior_diffs = order*np.eye(dim,dtype=int)
  p = _penalty(cutoff,order,sigma)
  L = rbf.fd.weight_matrix(x,x,prior_diffs,**kwargs)
  W = _diag(1.0/sigma)

  lhs = W.T.dot(W) + L.T.dot(L)/p**2
  rhs = W.T.dot(W).dot(u[i,~mask])
