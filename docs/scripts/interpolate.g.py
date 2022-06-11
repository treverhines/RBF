'''
Interpolating noisy scattered observations of Franke's test function with
smoothed RBFs.
'''
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
np.random.seed(1)

def frankes_test_function(x):
    '''A commonly used test function for interpolation'''
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y

# observation points
x_obs = np.random.random((100, 2))
# noisy values at the observation points
u_obs = frankes_test_function(x_obs) + np.random.normal(0.0, 0.1, (100,))
# evaluation points
x_itp = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T

# automatically select a smoothing parameter that minimizes the leave-one-out
# cross validation (LOOCV) score.
interp = RBFInterpolant(x_obs, u_obs, sigma='auto')
# interpolated values at the evaluation points
u_itp = interp(x_itp)

# We could also minimize the LOOCV score ourselves with the `loocv` method.
# This may be helpful for better exploring the objective function.
sigmas = 10**np.linspace(-2.5, 0.5, 100)
loocvs = [interp.loocv(x_obs, u_obs, sigma=s) for s in sigmas]
opt_sigma = sigmas[np.argmin(loocvs)]
print('optimal sigma: %s' % opt_sigma)

# plot the results
plt.figure(1)
plt.tripcolor(
    x_itp[:, 0], x_itp[:, 1], u_itp, cmap='viridis', vmin=0.0, vmax=1.2
    )
plt.scatter(
    x_obs[:, 0], x_obs[:, 1], s=100, c=u_obs, cmap='viridis', vmin=0.0,
    vmax=1.2, edgecolor='k'
    )
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.colorbar()
plt.title('Noisy observations and smoothed interpolant')
plt.tight_layout()
plt.savefig('../figures/interpolate.g.1.png')

plt.figure(2)
plt.loglog(sigmas, loocvs, 'k-')
plt.plot([opt_sigma], [np.min(loocvs)], 'ko')
plt.grid()
plt.xlabel('smoothing parameter')
plt.ylabel('LOOCV')
plt.tight_layout()
plt.savefig('../figures/interpolate.g.2.png')
plt.show()
