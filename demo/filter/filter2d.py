#!/usr/bin/env python
# this function demonstrates rbf.filter.filter for two-dimensional 
# data
import numpy as np
import matplotlib.pyplot as plt
import rbf.filter
import rbf.domain
import rbf.nodes
from scipy.signal import periodogram
import logging
np.random.seed(1)
logging.basicConfig(level=logging.DEBUG)

# number of observations
N = 1000

# observation points will be within the unit circle
vert,smp = rbf.domain.circle()
x,sid = rbf.nodes.menodes(N,vert,smp)

# the observations are a sin wave with added white noise
obs_std = np.ones(N)
obs_mean = np.random.normal(0.0,obs_std) + np.sin(2*np.pi*x[:,0])

# cutoff frequency
cutoff = 2.0

# derivative order
order = 2
post_mean,post_std = rbf.filter.filter(x,obs_mean,sigma=obs_std,
                                       order=order,cutoff=cutoff)

# plot the results
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax1 = ax[0]
ax2 = ax[1]
p = ax1.tripcolor(x[:,0],x[:,1],obs_mean)
ax1.plot(x[:,0],x[:,1],'k.')
plt.colorbar(p,ax=ax1)
p = ax2.tripcolor(x[:,0],x[:,1],post_mean)
ax2.plot(x[:,0],x[:,1],'k.')
plt.colorbar(p,ax=ax2)

ax1.grid(True)
ax1.set_title(u'$\mathrm{observed}$')
ax1.set_aspect('equal')
ax2.grid(True)
ax2.set_title(u'$\mathrm{filtered},\ \omega_c=2.0$')
ax2.set_aspect('equal')
fig.tight_layout()
plt.savefig('figures/filter2d.png')
plt.show()



