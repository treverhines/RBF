'''
This script demonstrates how to define a 1D Gibb Gaussian process
which has variable lengthscales. 
'''
import rbf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def lengthscale(x):
    # define an arbitrary lengthscale function 
    out = 0.25 + 0.5*np.abs(x)
    return out


gp = rbf.gauss.gpgibbs(lengthscale,0.5,delta=1e-3)

# define some points where we will evaluate the Gaussian process
x = np.linspace(-5,5,600)[:,None]

# get the mean and std. dev. of the Gaussian process
mean,std = gp(x)

# draw some samples of the process
u1 = gp.sample(x)
u2 = gp.sample(x)
u3 = gp.sample(x)

# plot the samples, and the lengthscale
fig,ax = plt.subplots()
ax.plot(x[:,0],lengthscale(x)[:,0],'k-',label='lengthscale')
ax.plot(x[:,0],mean,'C0-',label='mean')
ax.fill_between(x[:,0], mean-std, mean+std, 
               color='C0', alpha=0.3, label='std. dev.')
ax.plot(x[:,0],u1,'C0--',label='sample 1')
ax.plot(x[:,0],u2,'C1--',label='sample 2')
ax.plot(x[:,0],u3,'C2--',label='sample 3')
ax.legend()
ax.set_xlim(-5.0,5.0)
ax.set_ylim(-3.0,3.0)
ax.grid(ls=':')
fig.tight_layout()
plt.show()

