''' 
This script describes how to use the *outliers* method to detect and
remove outliers prior to conditioning a *GaussinaProcess*. 
'''
import numpy as np
import matplotlib.pyplot as plt
import logging
from rbf.gauss import gpse, gppoly
logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

y = np.linspace(-7.5,7.5,50) # obsevation points
x = np.linspace(-7.5,7.5,1000) # interpolation points
truth = np.exp(-0.3*np.abs(x))*np.sin(x)  # true signal at interp. points
# form synthetic data
obs_sigma = 0.1*np.ones(50) # noise standard deviation
noise = np.random.normal(0.0,obs_sigma)
noise[20],noise[25] = 2.0,1.0 # add anomalously large noise
obs_mu = np.exp(-0.3*np.abs(y))*np.sin(y) + noise
# form prior Gaussian process
prior = gpse((0.0, 1.0, 1.0)) + gppoly(1)
# find outliers which will be removed
toss = prior.outliers(y[:,None],obs_mu,obs_sigma)
# condition with non-outliers
post = prior.condition(y[~toss,None],obs_mu[~toss],obs_sigma[~toss])
post_mu,post_sigma = post(x[:,None])
# plot the results
fig,ax = plt.subplots(figsize=(6,4)) 
ax.errorbar(y,obs_mu,obs_sigma,fmt='k.',capsize=0.0,label='observations') 
ax.plot(x,post_mu,'b-',label='posterior mean') 
ax.fill_between(x,post_mu-post_sigma,post_mu+post_sigma,
                color='b',alpha=0.2,edgecolor='none',
                label='posterior uncertainty') 
ax.plot(x,truth,'k-',label='true signal') 
ax.legend(fontsize=10)
ax.set_xlim((-7.5,7.5))
ax.grid(True)
fig.tight_layout()
plt.savefig('../figures/gauss.c.png')
plt.show()

