'''
RBF Interpolation on a unit sphere. This is done by converting theta
(azimuthal angle) and phi (polar angle) into cartesian coordinates and
then interpolating over R^3.
'''
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
np.random.seed(1)

def spherical_to_cartesian(theta, phi):
    x = np.sin(phi)*np.cos(theta)  
    y = np.sin(phi)*np.sin(theta)  
    z = np.cos(phi)
    return np.array([x, y, z]).T


def true_function(theta, phi):
  # create some arbitary function that we want to recontruct with
  # interpolation
  cart = spherical_to_cartesian(theta, phi)
  out = (np.cos(cart[:, 2] - 1.0)*
         np.cos(cart[:, 0] - 1.0)*
         np.cos(cart[:, 1] - 1.0))
  return out

# number of observation points    
nobs = 50
# make the observation points in spherical coordinates 
theta_obs = np.random.uniform(0.0, 2*np.pi, nobs)
phi_obs = np.random.uniform(0.0, np.pi, nobs)
# get the cartesian coordinates for the observation points
cart_obs = spherical_to_cartesian(theta_obs, phi_obs)
# get the latent function at the observation points
val_obs = true_function(theta_obs, phi_obs)
# make a grid of interpolation points in spherical coordinates
theta_itp, phi_itp = np.meshgrid(np.linspace(0.0, 2*np.pi, 100),
                                 np.linspace(0.0, np.pi, 100))
theta_itp = theta_itp.flatten()
phi_itp = phi_itp.flatten()
# get the catesian coordinates for the interpolation points
cart_itp = spherical_to_cartesian(theta_itp, phi_itp)
# create an RBF interpolant from the cartesian observation points. I
# am just use the default `RBFInterpolant` parameters here, nothing
# special.
I = RBFInterpolant(cart_obs, val_obs)
# evaluate the interpolant on the interpolation points
val_itp = I(cart_itp)

## PLOTTING

# plot the true function in spherical coordinates
val_true = true_function(theta_itp, phi_itp)
plt.figure(1)
plt.title('True function')
p = plt.tripcolor(theta_itp, phi_itp, val_true,
                  cmap='viridis')
plt.colorbar(p)
plt.xlabel('theta (azimuthal angle)')
plt.ylabel('phi (polar angle)')
plt.xlim(0, 2*np.pi)
plt.ylim(0, np.pi)
plt.grid(ls=':', color='k')
plt.tight_layout()
plt.savefig('../figures/interpolate.b.1.png')

# plot the interpolant in spherical coordinates
plt.figure(2)
plt.title('RBF interpolant (points are observations)')
# plot the interpolated function
p = plt.tripcolor(theta_itp, phi_itp, val_itp,
                  cmap='viridis')
# plot the observations
plt.scatter(theta_obs, phi_obs, c=val_obs,
            s=50, edgecolor='k', cmap='viridis',
            vmin=p.get_clim()[0], vmax=p.get_clim()[1])                  
plt.colorbar(p)
plt.xlabel('theta (azimuthal angle)')
plt.ylabel('phi (polar angle)')
plt.xlim(0, 2*np.pi)
plt.ylim(0, np.pi)
plt.grid()
plt.grid(ls=':', color='k')
plt.tight_layout()
plt.savefig('../figures/interpolate.b.2.png')

# compute and print the mean L2 error
mean_error = np.mean(np.abs(val_true - val_itp))
print('mean interpolation error: %s' % mean_error)
plt.show()
