''' 
This script defines a function which performs robust Gaussian process 
regression, where outliers are detected and ignored. This algorithm is 
iterative; outliers are detected based on their misfit to the 
posterior Gaussian process, and then a new posterior Gaussian process 
is calculated with the detected outliers removed. Iterations continue 
until no new outliers are detected. This algorithm is based on the 
data editing algorithm from [1].

References
----------
[1] Gibbs, B. Advanced Kalman Filtering, Least-Squares and 
  Modeling: A Practical Handbook. John Wiley & Sons, 2011.

'''
import numpy as np
import matplotlib.pyplot as plt
import rbf
np.random.seed(1)

def robust_gpr(y,d,s,coeffs,x=None,basis=rbf.basis.se,order=1,tol=3.0):
  ''' 
  Parameters
  ----------
  y : (N,D) array
    observation points

  d : (N,) array
    observations

  s : (N,) array
    observation uncertainties    
  
  coeffs : (3,) tuple
    mean, variance, and characteristic length-scale of the prior 
    Gaussian process
  
  x : (M,D) array
    output points. Defaults to *y*
    
  basis : rbf.basis.RBF instance, optional
    prior covariance function

  order : int, optional  
    order of the polynomial null space
  
  tol : float, optional
    outlier tolerance. Smaller values make the outlier detection 
    algorithm more sensitive.
    
  Returns 
  -------
  out_mean : (M,) array
  out_sigma : (M,) array
  
  '''
  if x is None: x = y
  # Initially assume that none of the data are outliers
  is_outlier = np.zeros(d.shape[0],dtype=bool)
  prior = rbf.gauss.gpiso(basis,coeffs) + rbf.gauss.gppoly(order)
  while True:
    # form posterior ignoring detected outliers
    post = prior.condition(y[~is_outlier],d[~is_outlier],sigma=s[~is_outlier])
    res = np.abs(post.mean(y) - d)/s # weighted residuals
    rms = np.sqrt(np.mean(res[~is_outlier]**2)) # root mean square error
    if np.all(is_outlier == (res > tol*rms)):
      # break out out while loop if no new outliers are detected
      print('detected outliers : %s' % is_outlier.nonzero()[0])
      break
    else:
      # outliers are those which have a weighted residual that is
      # significantly greater than the root mean square error
      is_outlier = (res > tol*rms)

  out_mean,out_sigma = post(x)
  return out_mean,out_sigma


y = np.linspace(-7.5,7.5,50) # obsevation points
x = np.linspace(-7.5,7.5,1000) # interpolation points
s = 0.1*np.ones(50) # noise standard deviation
noise = np.random.normal(0.0,s)
noise[20],noise[25] = 2.0,1.0 # add anomalously large noise
d_true = np.exp(-0.3*np.abs(x))*np.sin(x)  # true signal at interp. points
d = np.exp(-0.3*np.abs(y))*np.sin(y) + noise  # observations
# find the mean and uncertainty of the posterior
u,us = robust_gpr(y[:,None],d,s,(0.0,100.0,2.0),x=x[:,None]) 
# plot the results
fig,ax = plt.subplots() 
ax.errorbar(y,d,s,fmt='k.',capsize=0.0,label='observations') 
ax.plot(x,u,'b-',label='posterior mean') 
ax.fill_between(x,u-us,u+us,color='b',alpha=0.2,edgecolor='none',label='posterior uncertainty') 
ax.plot(x,d_true,'k-',label='true signal') 
ax.legend(frameon=False,fontsize=10)
ax.set_xlim((-7.5,7.5))
ax.grid(True)
fig.tight_layout()
plt.savefig('../figures/gauss.c.png')
plt.show()

