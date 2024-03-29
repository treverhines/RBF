''' 
This script demonstrates how create a Gaussian process describing 
Brownian motion with the *GaussianProcess* class. Brownian motion has 
mean u(t) = 0 and covariance cov(t1,t2) = min(t1,t2).
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.gproc import GaussianProcess
np.random.seed(1)

# define mean function
def brownian_mean(x):
  out = np.zeros(x.shape[0])
  return out

# define covariance function
def brownian_cov(x1,x2):
  c = np.min(np.meshgrid(x1[:,0], x2[:,0], indexing='ij'), axis=0)
  return c

t = np.linspace(0.001, 1, 500)[:,None]
brown = GaussianProcess(mean=brownian_mean, covariance=brownian_cov)
sample = brown.sample(t) # draw a sample
mu,sigma = brown(t) # evaluate the mean and std. dev.

# plot the results
fig,ax = plt.subplots(figsize=(6, 4))
ax.grid(True)
ax.plot(t[:,0], mu, 'b-', label='mean')
ax.fill_between(t[:,0], mu + sigma, mu - sigma, color='b', alpha=0.2, edgecolor='none', label='std. dev.')
ax.plot(t[:,0],sample,'k',label='sample')
ax.set_xlabel('time',fontsize=10)
ax.set_title('Brownian motion',fontsize=10)
ax.set_xlim((0,1))
ax.tick_params(labelsize=10)
fig.tight_layout()
plt.savefig('../figures/gproc.d.png')
plt.show()

