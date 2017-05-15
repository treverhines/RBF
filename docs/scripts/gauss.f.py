''' 
This script demonstrate how to optimize the hyperparameters for a 
Gaussian process based on the marginal likelihood. Optimization is 
performed in two ways, first with a grid search method and then with a 
downhill simplex method.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import rbf
np.random.seed(3)

# True signal which we want to recover. This is a exponential function
# with mean=0.0, variance=2.0, and time-scale=0.1. For graphical
# purposes, we will only estimate the variance and time-scale.
a,b,c = 0.0,2.0,0.1
gp  = rbf.gauss.gpexp((a,b,c))

n = 500 # number of observations
time = np.linspace(-5.0,5.0,n)[:,None] # observation points
data = gp.sample(time) # signal which we want to describe

# find the optimal hyperparameter with a brute force grid search
b_search = 10**np.linspace(-2,2,30)
c_search = 10**np.linspace(-2,1,30)
likelihoods = np.zeros((30,30))
for i,b_test in enumerate(b_search): 
  for j,c_test in enumerate(c_search): 
    gp = rbf.gauss.gpexp((0.0,b_test,c_test))
    likelihoods[i,j] = gp.likelihood(time,data)

# find the optimal hyperparameters with a positively constrained 
# downhill simplex method
def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint''' 
  def pos_func(x,*blargs):
    return func(np.exp(x),*blargs)

  out = fmin(pos_func,np.log(x0),*args,**kwargs)
  out = np.exp(out)
  return out

def objective(x,t,d):
  '''objective function to be minimized'''
  gp  = rbf.gauss.gpexp((0.0,x[0],x[1]))
  return -gp.likelihood(t,d)

# maximum likelihood estimate
b_mle,c_mle = fmin_pos(objective,[1.0,1.0],args=(time,data))

# plot the results
fig,axs = plt.subplots(2,1,figsize=(6,6))
ax = axs[0]
ax.grid(True)
ax.plot(time[:,0],data,'k.',label='observations')
ax.set_xlim((-5.0,5.0))
ax.set_ylim((-5.0,7.0))
ax.set_xlabel('time',fontsize=10)
ax.set_ylabel('observations',fontsize=10)
ax.tick_params(labelsize=10)
ax.text(0.55,0.925,r"$\bar{u}(x) = a$",transform=ax.transAxes)
ax.text(0.55,0.85,r"$C_u(x,x') = b\times\exp(-|x - x'|/c)$",transform=ax.transAxes)
ax.text(0.55,0.775,r"$a = %s$, $b = %s$, $c = %s$" % (a,b,c),transform=ax.transAxes)
           
ax = axs[1]
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('variance ($b$)',fontsize=10)
ax.set_ylabel('time-scale ($c$)',fontsize=10)
ax.tick_params(labelsize=10)
p = ax.contourf(b_search,c_search,likelihoods.T,np.linspace(-5000.0,-200.0,11),cmap='viridis')
cbar = plt.colorbar(p,ax=ax)
ax.plot([b],[c],'ro',markersize=10,mec='none',label='truth')
ax.plot([b_mle],[c_mle],'ko',markersize=10,mec='none',label='max likelihood')
ax.legend(fontsize=10,loc=2)
cbar.set_label('log likelihood',fontsize=10)
cbar.ax.tick_params(labelsize=10)
ax.grid(True)
plt.tight_layout()
plt.savefig('../figures/gauss.f.png')
#plt.show()
