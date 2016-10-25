''' 
This script demonstrates smoothing and interpolating two-dimensional 
data when there is a known discontinuity in the underlying signal.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.filter import filter
np.random.seed(1)

def signal(x):
  ''' 
  this signal has a discontinuity from (0.0,-2.0) to (0.0,2.0)
  '''
  a = np.arctan2(x[:,0], 2 + x[:,1])
  b = np.arctan2(x[:,0],-2 + x[:,1])
  return (a - b)/np.pi

# define the known discontinuity
bnd_vert = np.array([[0.0, 2.0],[0.0,-2.0]])
bnd_smp = np.array([[0,1]])                     

N = 200 # number of observations
pnts_obs = np.random.uniform(-4,4,(N,2))
# create observations with added white noise
u_obs = signal(pnts_obs) + np.random.normal(0.0,0.1,N)
sigma_obs = 0.1*np.ones(N)

# create the interpolation points
vals = np.linspace(-4,4,100) 
pnts_itp = np.reshape(np.meshgrid(vals,vals),(2,10000)).T

# rbf.filter.filter can be used for interpolation by treating the 
# interpolation points as observation points with infinite 
# uncertainty. This is done by concatenating pnts_obs and pnts_itp and 
# then appending observations with infinite uncertainty to u_obs
pnts = np.concatenate((pnts_obs,pnts_itp),axis=0)

u_itp = np.empty(10000) + np.nan
sigma_itp = np.empty(10000) + np.inf

u = np.concatenate((u_obs,u_itp))
sigma = np.concatenate((sigma_obs,sigma_itp))

# cutoff frequency. Because we are interpolating data, there will not 
# be a homogeneous characteristic wavelength for the solution. 
# Interpolation points which do not have nearby observations will tend 
# to have a longer wavelength.  The cutoff frequency should be 
# interpretted as a rough estimate of the highest frequency allowed in 
# the final solution.
cutoff = 0.2

soln = filter(pnts,u,sigma=sigma,cutoff=cutoff,vert=bnd_vert,smp=bnd_smp)

# plot the results
fig,axs = plt.subplots(2,1,figsize=(6,10))
p = axs[0].tripcolor(pnts[:,0],pnts[:,1],soln[0],vmin=-1.0,vmax=1.0,cmap='viridis')
axs[0].scatter(pnts_obs[:,0],pnts_obs[:,1],c=u_obs,s=50,vmin=-1.0,vmax=1.0,cmap='viridis')
axs[0].plot(bnd_vert[:,0],bnd_vert[:,1],'r-',lw=6)
axs[0].set_xlim((-4,4))
axs[0].set_ylim((-4,4))
axs[0].set_aspect('equal')
axs[0].set_title(u'Observed and Filtered Solution. $(\omega_c=%s)$' % cutoff)
plt.colorbar(p,ax=axs[0])
p = axs[1].tripcolor(pnts[:,0],pnts[:,1],signal(pnts),vmin=-1.0,vmax=1.0,cmap='viridis')
axs[1].plot(bnd_vert[:,0],bnd_vert[:,1],'r-',lw=6)
axs[1].set_xlim((-4,4))
axs[1].set_ylim((-4,4))
axs[1].set_title('True Signal')
axs[1].set_aspect('equal')
plt.colorbar(p,ax=axs[1])
#plt.savefig('../figures/filter.c.png')
plt.tight_layout()
plt.show()
