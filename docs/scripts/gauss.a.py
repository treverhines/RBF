''' 
This script demonstrates how to perform Gaussian process regression on a noisy
data set. It also demonstrates drawing samples of the prior and posterior to
provide the user with an intuitive understanding of their distributions.
'''
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from rbf.gauss import gpiso, gppoly

logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

# observation points
y = np.linspace(-7.5, 7.5, 25) 
# interpolation points
x = np.linspace(-7.5, 7.5, 1000) 
# true signal
u_true = np.exp(-0.3*np.abs(x))*np.sin(x)
# observation uncertainty
sigma = 0.1*np.ones(25) 
# noisy observations of the signal
d = np.exp(-0.3*np.abs(y))*np.sin(y) + np.random.normal(0.0, sigma)
# form a prior Gaussian process which has a squared exponential covariance
# function (rbf.basis.se), 0.0 for the mean, and variance and lengthscales that
# are chosen with maximum likelihood. The variance and lengthscale are given
# positivity contraints by optimizing them in log space
def negative_likelihood(params):
    log_variance, log_lengthscale = params
    gp = gpiso('se', (0.0, 10**log_variance, 10**log_lengthscale)) 
    return -gp.likelihood(y[:, None], d, sigma=sigma)

log_variance, log_lengthscale = minimize(negative_likelihood, [0.0, 0.0]).x
# create a prior GaussianProcess using the most likely variance and lengthscale
gp_prior = gpiso('se', (0.0, 10**log_variance, 10**log_lengthscale)) 
# generate a sample of the prior
sample_prior = gp_prior.sample(x[:, None]) 
# find the mean and standard deviation of the prior
mean_prior, std_prior = gp_prior(x[:, None])
# condition the prior on the observations
gp_post = gp_prior.condition(y[:, None], d, sigma=sigma) 
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
ax.errorbar(y, d, sigma, fmt='ko', capsize=0, label='observations')
ax.plot(x, u_true, 'k-', label='true signal')
ax.plot(x, mean_post, 'b-', label='mean')
ax.plot(x, sample_post, 'b--', label='sample')
ax.fill_between(x, mean_post-std_post, mean_post+std_post, color='b',
                alpha=0.2, edgecolor='none', label='standard deviation')
ax.set_xlim((-7.5, 7.5))
ax.set_ylim((-0.75, 1.0))
ax.legend(loc=2, fontsize=10)
plt.tight_layout()
plt.savefig('../figures/gauss.a.png')
plt.show()
