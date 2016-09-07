#!/usr/bin/env python
# this function demonstrates rbf.smooth.smooth for one-dimensional 
# data and shows that the smoothing function is effectively equivalent 
# to a Butterworth low-pass filter
import numpy as np
import matplotlib.pyplot as plt
import rbf.smooth
from scipy.signal import periodogram
import logging
np.random.seed(1)
logging.basicConfig(level=logging.DEBUG)

# number of observations
N = 10000
t = np.linspace(0.0,N,N)[:,None]

# the observations are just white noise
obs_std = np.ones(N)
obs_mean = np.random.normal(0.0,obs_std)

# cutoff frequency
cutoff = 0.01

# derivative order
order = 2

post_mean,post_std = rbf.smooth.smooth(t,obs_mean,sigma=obs_std,
                                       order=order,cutoff=cutoff)

# plot the results
fig,ax = plt.subplots(1,2,figsize=(8,4))
ax1 = ax[0]
ax2 = ax[1]
ax1.plot(t[:,0],obs_mean,'k.')
ax1.plot(t[:,0],post_mean,'b-')
ax1.fill_between(t[:,0],post_mean-post_std,post_mean+post_std,
                 color='b',alpha=0.2,edgecolor='none')

# compute and plot the power spectral density of the observed and 
# smoothed data
freq,pow = periodogram(obs_mean,1.0/(t[1]-t[0]))
ax2.loglog(freq[1:],pow[1:],'k-')
freq,pow = periodogram(post_mean,1.0/(t[1]-t[0]))
ax2.loglog(freq[1:],pow[1:],'b-')
ylim = ax2.get_ylim()
ax2.set_ylim(ylim)
# plot the expected frequency content of the smoothed solution
ax2.loglog(freq[1:],(1.0/(1.0 + (freq[1:]/cutoff)**(2*order)))**2,'k--')

ax1.grid(True)
ax1.set_xlabel(u'$\mathrm{time}$')
ax1.set_ylim((-7,7))
ax1.legend([u'$\mathrm{observed}$',u'$\mathrm{smoothed},\ \omega_c=10^{-2}$'],frameon=False,fontsize=12,loc=0)
ax2.set_xlabel(u'$\mathrm{frequency}$')
ax2.set_ylabel(u'$\mathrm{power\ spectral\ density}$')
ax2.set_xlim((10**-4.5,1))
ax2.legend([u'$\mathrm{observed}$',u'$\mathrm{smoothed},\ \omega_c=10^{-2}$',u'$\mathrm{expected}$'],frameon=False,loc=0,fontsize=12)
ax2.grid(True)
fig.tight_layout()
plt.savefig('figures/filter1d.png')
plt.show()



