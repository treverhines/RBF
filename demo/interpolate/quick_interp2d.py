#!/usr/bin/env python
# This script demonstrates how to use RBFs for 2-D interpolation
import numpy as np
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
np.random.seed(1)

# create noisy data
x_obs = np.random.random((100,2)) # observation points
u_obs = np.sin(2*np.pi*x_obs[:,0])*np.cos(2*np.pi*x_obs[:,1])
u_obs += np.random.normal(0.0,0.2,100)

# create smoothed interpolant
I = RBFInterpolant(x_obs,u_obs,penalty=0.001)

# create interpolation points
x_itp = np.random.random((10000,2))
u_itp = I(x_itp)

plt.tripcolor(x_itp[:,0],x_itp[:,1],u_itp,vmin=-1.1,vmax=1.1)
plt.scatter(x_obs[:,0],x_obs[:,1],s=100,c=u_obs,vmin=-1.1,vmax=1.1)
plt.xlim((0.05,0.95))
plt.ylim((0.05,0.95))
plt.colorbar()
plt.tight_layout()
plt.show()
