''' 
This script demonstrates how to perform Gaussian process regression on a noisy
data set. It also demonstrates drawing samples of the prior and posterior to
provide the user with an intuitive understanding of their distributions.
'''
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from rbf.gproc import gpiso

logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

# observation points
y = np.linspace(-7.5, 7.5, 25) 
# interpolation points
x = np.linspace(-7.5, 7.5, 1000) 
# true signal
u_true = np.exp(-0.3*np.abs(x))*np.sin(x)
# observation noise covariance
dsigma = np.full(25, 0.1)
dcov = np.diag(dsigma**2)
# noisy observations of the signal
d = np.exp(-0.3*np.abs(y))*np.sin(y) + np.random.normal(0.0, dsigma)

def negative_log_likelihood(params):
    log_eps, log_var = params
    gp = gpiso('se', eps=10**log_eps, var=10**log_var) 
    out = -gp.log_likelihood(y[:, None], d, dcov=dcov)
    return out

log_eps, log_var = minimize(negative_log_likelihood, [0.0, 0.0]).x
# create a prior GaussianProcess using the most likely variance and lengthscale
gp_prior = gpiso('se', eps=10**log_eps, var=10**log_var)
# generate a sample of the prior
sample_prior = gp_prior.sample(x[:, None]) 
# find the mean and standard deviation of the prior
mean_prior, std_prior = gp_prior(x[:, None])
# condition the prior on the observations
gp_post = gp_prior.condition(y[:, None], d, dcov=dcov)
sample_post = gp_post.sample(x[:, None]) 
mean_post, std_post = gp_post(x[:, None])

## Plotting
#####################################################################
fig, axs = plt.subplots(2, 1, figsize=(6, 6))
ax = axs[0]
ax.grid(ls=':')
ax.tick_params(labelsize=10)
ax.set_title('Prior Gaussian Process', fontsize=10)
ax.plot(x, mean_prior, 'b-', label='mean')
ax.fill_between(x, mean_prior - std_prior, mean_prior + std_prior, color='b',
                alpha=0.2, edgecolor='none', label='standard deviation')

ax.plot(x, sample_prior, 'b--', label='sample')
ax.set_xlim((-7.5, 7.5))
ax.set_ylim((-2.0, 2.0))
ax.legend(loc=2, fontsize=10)
ax = axs[1]
ax.grid(ls=':')
ax.tick_params(labelsize=10)
ax.set_title('Conditioned Gaussian Process', fontsize=10)
ax.errorbar(y, d, dsigma, fmt='ko', capsize=0, label='observations')
ax.plot(x, u_true, 'k-', label='true signal')
ax.plot(x, mean_post, 'b-', label='mean')
ax.plot(x, sample_post, 'b--', label='sample')
ax.fill_between(x, mean_post-std_post, mean_post+std_post, color='b',
                alpha=0.2, edgecolor='none', label='standard deviation')
ax.set_xlim((-7.5, 7.5))
ax.set_ylim((-0.75, 1.0))
ax.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.savefig('../figures/gproc.a.png')
plt.show()
