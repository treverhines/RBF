#!/usr/bin/env python
# this function demonstrates using rbf.filter.filter to differentiate 
# a one-dimensional time series
import numpy as np
import matplotlib.pyplot as plt
import rbf.filter
from scipy.signal import periodogram
import logging
np.random.seed(1)
logging.basicConfig(level=logging.DEBUG)

# number of observations
N = 1000
t = np.linspace(0.0,10.0,N)[:,None]

# the observations are a sin wave white noise 
obs_std = np.ones(N)
obs_mean = np.random.normal(0.0,obs_std) + np.sin(2*np.pi*t[:,0])

# cutoff frequency
cutoff = 2.0

# derivative order
order = 2
post_mean,post_std = rbf.filter.filter(t,obs_mean,sigma=obs_std,
                                       order=order,cutoff=cutoff,samples=1000)
diff_mean,diff_std = rbf.filter.filter(t,obs_mean,sigma=obs_std,diffs=(1,),
                                       order=order,cutoff=cutoff,samples=1000)

# plot the results
fig,ax = plt.subplots(2,1,figsize=(8,6))
ax1 = ax[0]
ax2 = ax[1]
ax1.plot(t[:,0],obs_mean,'k.')
ax1.plot(t[:,0],post_mean,'b-')
ax1.fill_between(t[:,0],post_mean-post_std,post_mean+post_std,
                 color='b',alpha=0.2,edgecolor='none')
ax1.plot(t[:,0],np.sin(2*np.pi*t[:,0]),'r-')
ax1.set_ylim((-6,8))
ax1.grid(True)
ax1.set_xlabel(u'$\mathrm{time}$')
ax1.legend([u'$\mathrm{observed}$',u'$\mathrm{filtered}$',u'$\mathrm{true\ signal}$'],
           frameon=False,fontsize=12,loc=1)

# plot differentiated solution
ax2.plot(t[:,0],diff_mean,'b-')
ax2.fill_between(t[:,0],
                 diff_mean-diff_std,
                 diff_mean+diff_std,
                 color='b',alpha=0.2,edgecolor='none')
ax2.plot(t[:,0],2*np.pi*np.cos(2*np.pi*t[:,0]),'r-')
ax2.legend([u'$\mathrm{filtered\ derivative}$',u'$\mathrm{true\ signal\ derivative}$'],
           frameon=False,fontsize=12,loc=1)
ax2.set_ylim((-10,15))
ax2.grid(True)
ax2.set_xlabel(u'$\mathrm{time}$')
fig.tight_layout()
plt.savefig('figures/diff1d.png')
plt.show()



