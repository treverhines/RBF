''' 
This script demonstrate how to optimize the hyperparameters for a 
Gaussian process based on the marginal likelihood. Optimization is 
performed in two ways, first with a grid search method and then with a 
downhill simplex method.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from rbf.gproc import gpiso
np.random.seed(3)

# True signal which we want to recover. This is a exponential function
# with mean=0.0, variance=2.0, and time-scale=0.1. For graphical
# purposes, we will only estimate the variance and time-scale.
eps = 0.1
var = 2.0
gp = gpiso('exp', eps=eps, var=var)

n = 500 # number of observations
time = np.linspace(-5.0, 5.0, n)[:,None] # observation points
data = gp.sample(time) # signal which we want to describe

# find the optimal hyperparameter with a brute force grid search
eps_search = 10**np.linspace(-2, 1, 30)
var_search = 10**np.linspace(-2, 2, 30)
log_likelihoods = np.zeros((30, 30))
for i, eps_test in enumerate(eps_search): 
  for j, var_test in enumerate(var_search): 
    gp = gpiso('exp', eps=eps_test, var=var_test)
    log_likelihoods[i, j] = gp.log_likelihood(time, data)

# find the optimal hyperparameters with a positively constrained 
# downhill simplex method
def fmin_pos(func, x0, *args, **kwargs):
  '''fmin with positivity constraint''' 
  def pos_func(x, *blargs):
    return func(np.exp(x), *blargs)

  out = fmin(pos_func, np.log(x0), *args, **kwargs)
  out = np.exp(out)
  return out

def objective(x, t, d):
  '''objective function to be minimized'''
  gp = gpiso('exp', eps=x[0], var=x[1])
  return -gp.log_likelihood(t,d)

# maximum likelihood estimate
eps_mle, var_mle = fmin_pos(objective, [1.0, 1.0], args=(time, data))

# plot the results
fig, axs = plt.subplots(2, 1, figsize=(6, 6))
ax = axs[0]
ax.grid(True)
ax.plot(time[:,0], data, 'k.', label='observations')
ax.set_xlim((-5.0, 5.0))
ax.set_ylim((-5.0, 7.0))
ax.set_xlabel('time', fontsize=10)
ax.set_ylabel('observations', fontsize=10)
ax.tick_params(labelsize=10)
ax.text(0.55, 0.85, r"$C_u(x,x') = \sigma^2\exp(-|x - x'|/\epsilon)$", transform=ax.transAxes)
ax.text(0.55, 0.775, r"$\sigma^2 = %s$, $\epsilon = %s$" % (var, eps), transform=ax.transAxes)
           
ax = axs[1]
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$\epsilon$', fontsize=10)
ax.set_ylabel(r'$\sigma^2$', fontsize=10)
ax.tick_params(labelsize=10)
p = ax.contourf(eps_search, var_search, log_likelihoods.T, np.linspace(-5000.0, -200.0, 11), cmap='viridis')
cbar = plt.colorbar(p, ax=ax)
ax.plot([eps], [var], 'ro', markersize=10, mec='none', label='truth')
ax.plot([eps_mle], [var_mle], 'ko', markersize=10, mec='none', label='max likelihood')
ax.legend(fontsize=10, loc=2)
cbar.set_label('log likelihood', fontsize=10)
cbar.ax.tick_params(labelsize=10)
ax.grid(True)
plt.tight_layout()
plt.savefig('../figures/gproc.f.png')
plt.show()
