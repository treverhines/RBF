''' 
This script demonstrates how create a Gaussian process describing 
integrated Brownian motion with the *GaussianProcess* class. 
integrated Brownian motion has mean u(t) = 0 and covariance 

  cov(t1, t2) = (1/2)*min(t1, t2)**2 * (max(t1, t2) - min(t1, t2)/3).

It also demonstrates how to make the *GaussianProcess* differentiable 
by manually specifying the covariance derivatives.

'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.gproc import GaussianProcess
np.random.seed(1)

# define mean function
def mean(x, diff):
  '''mean function which is zero for all derivatives'''
  out = np.zeros(x.shape[0])
  return out

# define covariance function
def cov(x1, x2, diff1, diff2):
  '''covariance function and its derivatives'''
  x1, x2 = np.meshgrid(x1[:, 0], x2[:, 0], indexing='ij')
  if (diff1 == (0,)) & (diff2 == (0,)):
    # integrated brownian motion
    out = (0.5*np.min([x1, x2], axis=0)**2*
           (np.max([x1, x2], axis=0) - 
            np.min([x1, x2], axis=0)/3.0))

  elif (diff1 == (1,)) & (diff2 == (1,)):
    # brownian motion
    out = np.min([x1, x2], axis=0)
  
  elif (diff1 == (1,)) & (diff2 == (0,)):   
    # derivative w.r.t x1
    out = np.zeros_like(x1)
    idx1 = x1 >= x2
    idx2 = x1 <  x2
    out[idx1] = 0.5*x2[idx1]**2
    out[idx2] = x1[idx2]*x2[idx2] - 0.5*x1[idx2]**2

  elif (diff1 == (0,)) & (diff2 == (1,)):   
    # derivative w.r.t x2
    out = np.zeros_like(x1)
    idx1 = x2 >= x1
    idx2 = x2 <  x1
    out[idx1] = 0.5*x1[idx1]**2
    out[idx2] = x2[idx2]*x1[idx2] - 0.5*x2[idx2]**2
  
  else:
    raise ValueError(
      'The *GaussianProcess* is not sufficiently differentiable')
  
  return out

t = np.linspace(0.0, 1, 100)[:, None]
gp = GaussianProcess(mean, cov, dim=1, differentiable=True)
sample = gp.sample(t) # draw a sample
mu, sigma = gp(t) # evaluate the mean and std. dev.

# plot the results
fig, ax = plt.subplots(figsize=(6, 4))
ax.grid(True)
ax.plot(t[:, 0], mu, 'b-', label='mean')
ax.fill_between(t[:, 0], mu+sigma, mu-sigma, color='b', alpha=0.2, edgecolor='none', label='std. dev.')
ax.plot(t[:, 0], sample, 'k', label='sample')
ax.set_xlabel('time', fontsize=10)
ax.set_xlim((0, 1))
ax.set_title('integrated Brownian motion', fontsize=10)
ax.tick_params(labelsize=10)
fig.tight_layout()
plt.savefig('../figures/gproc.g.png')
plt.show()

