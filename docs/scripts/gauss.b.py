''' 
This script demonstrates how to make a custom *GaussianProcess* by 
combining *GaussianProcess* instances. The resulting Gaussian process 
has two distinct length-scales.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.basis import se
from rbf.gauss import gpiso
np.random.seed(1)

dx = np.linspace(0.0, 5.0, 1000)[:,None]
x = np.linspace(-5, 5.0, 1000)[:,None]
gp_long = gpiso(se, (0.0, 1.0, 2.0))
gp_short = gpiso(se, (0.0, 0.5, 0.25))
gp = gp_long + gp_short
# compute the autocovariances 
acov_long = gp_long.covariance(dx, [[0.0]])
acov_short = gp_short.covariance(dx, [[0.0]])
acov = gp.covariance(dx, [[0.0]])
# draw 3 samples
sample = gp.sample(x) 
# mean and uncertainty of the new gp
mean,sigma = gp(x)
# plot the autocovariance functions
fig,axs = plt.subplots(2,1,figsize=(6,6))
axs[0].plot(dx,acov_long,'r--',label='long component')
axs[0].plot(dx,acov_short,'b--',label='short component')
axs[0].plot(dx,acov,'k-',label='sum')
axs[0].set_xlabel('$\mathregular{\Delta x}$',fontsize=10)
axs[0].set_ylabel('auto-covariance',fontsize=10)
axs[0].legend(fontsize=10)
axs[0].tick_params(labelsize=10)
axs[0].set_xlim((0,4))
axs[0].grid(True)
# plot the samples
axs[1].plot(x,sample,'k--',label='sample') 
axs[1].plot(x,mean,'k-',label='mean')
axs[1].fill_between(x[:,0],mean-sigma,mean+sigma,color='k',alpha=0.2,edgecolor='none',label='std. dev.')
axs[1].set_xlabel('x',fontsize=10)
axs[1].legend(fontsize=10)
axs[1].tick_params(labelsize=10)
axs[1].set_xlim((-5,5))
axs[1].grid(True)
plt.tight_layout()
plt.savefig('../figures/gauss.b.png')
plt.show()
