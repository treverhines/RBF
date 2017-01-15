#!/usr/bin/env python
# This script demonstrates how to use RBFs for 2-D interpolation
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
from rbf.interpolate import RBFInterpolant
import matplotlib.pyplot as plt
from matplotlib import cm
# set default cmap to viridis if you have it
if 'viridis' in vars(cm):
  plt.rcParams['image.cmap'] = 'viridis'

np.random.seed(1)

# create 20 2-D observation points
x = np.random.random((100,2))
# find the function value at the observation points
u = np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
u += np.random.normal(0.0,0.1,100)
# create interpolation points
a = np.linspace(0,1,100)
x1itp,x2itp = np.meshgrid(a,a)
xitp = np.array([x1itp.ravel(),x2itp.ravel()]).T
# form interpolant
I = RBFInterpolant(x,u,penalty=0.001)
# evaluate the interpolant
uitp = I(xitp)
# evaluate the x derivative of the interpolant
dxitp = I(xitp,diff=(1,0))

# find the true values
utrue = np.sin(2*np.pi*xitp[:,0])*np.cos(2*np.pi*xitp[:,1])
dxtrue = 2*np.pi*np.cos(2*np.pi*xitp[:,0])*np.cos(2*np.pi*xitp[:,1])

# plot the results
fig,ax = plt.subplots(2,2)
p = ax[0,0].tripcolor(xitp[:,0],xitp[:,1],uitp)
ax[0,0].scatter(x[:,0],x[:,1],c=u,s=100,clim=p.get_clim())
fig.colorbar(p,ax=ax[0,0])
p = ax[0,1].tripcolor(xitp[:,0],xitp[:,1],dxitp)
fig.colorbar(p,ax=ax[0,1])
p = ax[1,0].tripcolor(xitp[:,0],xitp[:,1],utrue)
fig.colorbar(p,ax=ax[1,0])
p = ax[1,1].tripcolor(xitp[:,0],xitp[:,1],dxtrue)
fig.colorbar(p,ax=ax[1,1])

ax[0,0].set_aspect('equal')
ax[1,0].set_aspect('equal')
ax[0,1].set_aspect('equal')
ax[1,1].set_aspect('equal')
ax[0][0].set_xlim((0,1))
ax[0][0].set_ylim((0,1))
ax[0][0].set_title('RBF interpolant')
ax[1][0].set_title('true solution')
ax[0][1].set_title('interpolant x derivative')
ax[1][1].set_title('true x derivative')
fig.tight_layout()
plt.savefig('figures/interp2d.png')
plt.show()
