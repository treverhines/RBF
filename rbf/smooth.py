''' 
provides a function for smoothing large, multidimensional, noisy data sets
'''   
import numpy as np
import rbf.fd
import rbf.basis
import scipy.sparse
from scipy.spatial import cKDTree
from rbf.interpolate import _in_hull

def mask(x,sigma,kind):
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


def smooth(x,value,sigma=None,
           cutoff=None, 
           fill='extrapolate',
           return_covariance=True,
           **kwargs):
  ''' 
  Parameters
  ----------
    x : (N,D) array
      observations points
    
    value : (...,N) array, 
      observations at x
    
    sigma : (N,) array, optional
      one standard deviation uncertainty on the observations  
    
    cutoff : float, optional
      cutoff frequency. Frequencies greater than this value will be 
      damped out
      
    order : int, optional
      polynomial order
      
    fill : str, optional
    
  Returns
  -------
    soln_mean : (...,N) array
    
    soln_cov : (N,N) array
  
  '''    
  x = np.asarray(x)
  value = np.asarray(value)  
  value = np.nan_to_num(value)
  value_shape = value.shape
  N,D = x.shape
  if sigma is None:
    sigma = np.ones(value.shape)

  if cutoff is None:
    cutoff = _default_cutoff(x)

  # throw out points where we do not want to estimate the solution
  keep_idx, = np.nonzero(~mask(x,sigma,fill))
  x = x[keep_idx]
  value = value[...,keep_idx]
  sigma = sigma[...,keep_idx]
  
  diff = 2*np.eye(D,dtype=int)
  if D == 1:
    L = rbf.fd.diff_matrix_1d(x,diff,**kwargs)
  else:
    L = rbf.fd.diff_matrix(x,diff,**kwargs)
    
  Cdata = 1.0/sigma**2
  K = len(x)
  Crow,Ccol = range(K),range(K)
  C = scipy.sparse.csr_matrix((Cdata,(Crow,Ccol)),(K,K))
  p = _penalty(cutoff,sigma)
  
  lhs = C + (1.0/p**2)*L.T.dot(L)
  lhs = lhs.tocsc()
  rhs = C.dot(value)

  soln_mean = scipy.sparse.linalg.spsolve(lhs,rhs)
  soln_mean_ext = np.empty(N) 
  soln_mean_ext[:] = np.nan
  soln_mean_ext[keep_idx] = soln_mean
  
  if return_covariance: 
    soln_cov = scipy.sparse.linalg.inv(lhs).toarray()
    soln_cov_ext = np.zeros((N,N))
    soln_cov_ext[np.ix_(keep_idx,keep_idx)] = soln_cov
  else:
    soln_cov_ext = None  
  
  return soln_mean_ext,soln_cov_ext
  

          
