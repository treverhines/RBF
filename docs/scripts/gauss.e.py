''' 
This script demonstrates how to create a periodic Gaussian process 
using the *RBFGaussianProcess* class.
'''
import numpy as np
import matplotlib.pyplot as plt
import rbf
from sympy import sin,exp,pi 
np.random.seed(1)

period = 5.0 
cls = 0.5 # characteristic length scale
var = 1.0 # variance

r = rbf.basis.get_r() # get symbolic variables
eps = rbf.basis.get_eps()
# create a symbolic expression of the periodic covariance function 
expr = exp(-sin(r*pi/period)**2/eps**2)
# define a periodic RBF using the symbolic expression
basis = rbf.basis.RBF(expr)
# define a Gaussian process using the periodic RBF
gp = rbf.gauss.RBFGaussianProcess((0.0,var,cls),basis=basis,dim=1)

t = np.linspace(-10,10,1000)[:,None]
sample = gp.draw_sample(t) # draw a sample
mu,sigma = gp(t) # evaluate mean and std. dev.

# plot the results
fig,ax = plt.subplots(figsize=(6,5))
ax.grid(True)
ax.plot(t[:,0],mu,'b-',label='mean')
ax.fill_between(t[:,0],mu-sigma,mu+sigma,color='b',alpha=0.2,edgecolor='none',label='std. dev.')
ax.plot(t,sample,'k',label='sample')
ax.set_xlim((-10.0,10.0))
ax.set_ylim((-2.5*var,2.5*var))
ax.legend(frameon=False,loc=4,fontsize=10)
ax.tick_params(labelsize=10)
ax.set_xlabel('time',fontsize=10)
ax.set_title('periodic Gaussian process',fontsize=10)
fig.tight_layout()
plt.savefig('../figures/gauss.e.png')
plt.show()

