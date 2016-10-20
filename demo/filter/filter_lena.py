#!/usr/bin/env python
# Use rbf.filer.filter to recover an underlying signal in a large 
# scattered data set. The underlying signal is the Lena image
import numpy as np
from scipy.misc import lena
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from rbf.filter import filter

x = np.linspace(0.0,1.0,512)
y = np.linspace(1.0,0.0,512)
x,y = np.meshgrid(x,y)
points = np.array([x.flatten(),y.flatten()]).T
values = lena().flatten().astype(float)
values /= 255.0 # normalize so that the max is 1.0
signal = NearestNDInterpolator(points,values)

# interpolate Lena onto new observation points and add noise
N = 100000
points_obs = np.random.normal(0.5,0.25,(N,2))
u_obs = signal(points_obs)
u_obs += np.random.normal(0.0,0.5,N)

# filter parameters
size = 20 # stencil size
cutoff = 40 # cutoff frequency

# find filtered solution
soln,sigma = filter(points_obs,u_obs,
                    cutoff=cutoff,size=size)

# plot the observed and filtered results
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_aspect('equal')
ax[0].set_xlim((0,1))
ax[0].set_ylim((0,1))
ax[0].set_title('Noisy Data')
ax[0].grid()
ax[1].set_aspect('equal')
ax[1].set_xlim((0,1))
ax[1].set_ylim((0,1))
ax[1].set_title(u'Filtered Solution $(\omega_c = %s)$' % cutoff)
ax[1].grid()
# plot the noisy scattered data 
p1 = ax[0].scatter(points_obs[:,0],points_obs[:,1],s=2,c=u_obs,
                   edgecolor='none',cmap='Greys_r',
                   vmin=-0.2,vmax=1.2)
plt.colorbar(p1,ax=ax[0])
# plot the filtered solution
p2 = ax[1].scatter(points_obs[:,0],points_obs[:,1],s=2,c=soln,
                   edgecolor='none',cmap='Greys_r',
                   vmin=-0.2,vmax=1.2)
plt.colorbar(p2,ax=ax[1])
plt.tight_layout()
plt.show()
