import numpy as np
import matplotlib.pyplot as plt
import rbf
np.random.seed(1)

y = np.linspace(-7.5,7.5,20) # observation points
x = np.linspace(-7.5,7.5,1000) # interpolation points
u_true = np.exp(-0.3*np.abs(x))*np.sin(x)  # true signal
sigma = 0.1*np.ones(20) # observation uncertainty
# noisy observations of the signal
d = np.exp(-0.3*np.abs(y))*np.sin(y) + np.random.normal(0.0,sigma) 
# form a prior Gaussian process which has a squared exponential basis 
# function (rbf.basis.ga), 0.0 for the mean, 1.0 for the standard 
# deviation, and 2.0 for the characteristic length scale.
gp = rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(1.0,0.0,2.0))
sample = gp.draw_sample(x[:,None]) # generate random sample
mean,std = gp(x[:,None]) # find the mean and standard dev. at x
gp_cond = gp.condition(y[:,None],d,sigma=sigma) # condition with data
sample_cond = gp_cond.draw_sample(x[:,None]) 
mean_cond,std_cond = gp_cond(x[:,None])  

## Plotting
#####################################################################
fig,axs = plt.subplots(2,1,figsize=(7,8))
ax = axs[0]
ax.tick_params(labelsize=10)
ax.set_title('Prior Gaussian Process',fontsize=10)
ax.plot(x,mean,'b-',label='mean')
ax.fill_between(x,mean-std,mean+std,color='b',
                alpha=0.2,edgecolor='none',label='standard deviation')
ax.plot(x,sample,'b--',label='sample')
ax.set_xlim((-7.5,7.5))
ax.set_ylim((-2.0,2.0))
ax.legend(loc=2,frameon=False,fontsize=10)
ax = axs[1]
ax.tick_params(labelsize=10)
ax.set_title('Conditioned Gaussian Process',fontsize=10)
ax.errorbar(y,d,sigma,fmt='ko',capsize=0,label='observations')
ax.plot(x,u_true,'k-',label='true signal')
ax.plot(x,mean_cond,'b-',label='mean')
ax.plot(x,sample_cond,'b--',label='sample')
ax.fill_between(x,mean_cond-std_cond,mean_cond+std_cond,color='b',
                alpha=0.2,edgecolor='none',label='standard deviation')
ax.set_xlim((-7.5,7.5))
ax.set_ylim((-0.75,1.0))
ax.legend(loc=2,frameon=False,fontsize=10)
plt.tight_layout()
plt.savefig('../figures/gpr.a.png')
plt.show()
