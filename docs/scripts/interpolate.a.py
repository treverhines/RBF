''' 
In this example we generate synthetic scattered data with added noise
and then fit it with a smoothed RBF interpolant. The interpolant in
this example is equivalent to a thin plate spline.
'''
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
np.random.seed(1)

# observation points
x_obs = np.random.random((100, 2)) 
# values at the observation points
u_obs = np.sin(2*np.pi*x_obs[:, 0])*np.cos(2*np.pi*x_obs[:, 1]) 
u_obs += np.random.normal(0.0, 0.1, 100)
# Create an RBF interpolant and specify a nonzero value for `sigma` to prevent
# fitting the noisy data exactly. By setting `phi` to "phs2", the interpolant
# is equivalent to a thin-plate spline.
sigma = minimize(lamdba s: RBFInterpolant.gml(x_obs, u_obs, sigma=s, phi='phs2'))
I = RBFInterpolant(x_obs, u_obs, sigma=0.1, eps=5.0, phi='ga')
# create a grid of evaluation points
x_itp = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T
u_itp = I(x_itp) 

# plot the results
plt.tripcolor(x_itp[:, 0], x_itp[:, 1], u_itp, vmin=-1.1, vmax=1.1, cmap='viridis')
plt.scatter(x_obs[:, 0], x_obs[:, 1], s=100, c=u_obs, vmin=-1.1, vmax=1.1,
            cmap='viridis', edgecolor='k')
plt.xlim((0.05, 0.95))
plt.ylim((0.05, 0.95))
plt.colorbar()
plt.tight_layout()
plt.savefig('../figures/interpolate.a.png')
plt.show()

