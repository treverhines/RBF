'''
Interpolating scattered observations of Franke's test function with RBFs.
'''
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
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
# values at the observation points
u_obs = frankes_test_function(x_obs)
# evaluation points
x_itp = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T

interp = RBFInterpolant(x_obs, u_obs)
# interpolated values at the evaluation points
u_itp = interp(x_itp)

# plot the results
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
plt.tight_layout()
plt.savefig('../figures/interpolate.f.png')
plt.show()
