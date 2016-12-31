import numpy as np
import matplotlib.pyplot as plt
import rbf
np.random.seed(1)

n = 20
# observation points
y = np.linspace(-7.5,7.5,n)
# interpolation points
x = np.linspace(-7.5,7.5,1000)
# true signal which we are trying to recover
u_true = np.exp(-0.3*np.abs(x))*np.sin(x)  
# observation uncertainty
sigma = 0.1*np.ones(n) 
# noisy observations of the signal
d = np.exp(-0.3*np.abs(y))*np.sin(y) + np.random.normal(0.0,sigma) 
# form a prior Gaussian process which has a squared exponential basis 
# function (rbf.basis.ga), 0.0 for the mean, 1.0 for the standard 
# deviation, and 2.0 for the characteristic length scale.
gp = rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(1.0,0.0,2.0))
# draw a sample from the Gaussian process
sample = gp.draw_sample(x[:,None])
# find the mean and standard deviation at x
mean,std = gp(x[:,None]) 
# plot the mean, standard deviation, and sample from the Gaussian 
# process
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
# Condition the Gaussian process with the observations
gp = gp.condition(y[:,None],d,sigma=sigma)
# uncomment the below line to make a new Gausisan process which is the
# first derivative of *gp* 
#gp = gp.differentiate((1,))
# draw a sample from the conditioned Gaussian process
sample = gp.draw_sample(x[:,None])
# find the mean and standard deviation at x
mean,std = gp(x[:,None]) 
# plot the mean, standard deviation, and sample from the conditioned 
# Gaussian process
ax = axs[1]
ax.tick_params(labelsize=10)
ax.set_title('Conditioned Gaussian Process',fontsize=10)
ax.errorbar(y,d,sigma,fmt='ko',capsize=0,label='observations')
ax.plot(x,u_true,'k-',label='true signal')
ax.plot(x,mean,'b-',label='mean')
ax.plot(x,sample,'b--',label='sample')
ax.fill_between(x,mean-std,mean+std,color='b',
                alpha=0.2,edgecolor='none',label='standard deviation')
ax.set_xlim((-7.5,7.5))
ax.set_ylim((-0.75,1.0))
ax.legend(loc=2,frameon=False,fontsize=10)
plt.tight_layout()
plt.savefig('../figures/gpr.a.png')
plt.show()
