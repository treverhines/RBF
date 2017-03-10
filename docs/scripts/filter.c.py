''' 
This script demonstrates smoothing 2-d data when there is a known 
discontinuity in the underlying signal.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.filter import filter
import rbf.halton
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
# create synthetic data
pnts_obs = 8*(rbf.halton.halton(200,2,start=1) - 0.5)
sigma_obs = 0.2*np.ones(200)
u_obs = signal(pnts_obs) + np.random.normal(0.0,sigma_obs)
# find the filtered solution
cutoff = 0.5
soln,_ = filter(pnts_obs,u_obs,sigma=sigma_obs,cutoff=cutoff,
                vert=bnd_vert,smp=bnd_smp)
# plot the results and true signal
vals = np.linspace(-4,4,200) 
grid = np.reshape(np.meshgrid(vals,vals),(2,200**2)).T
fig,axs = plt.subplots(2,1,figsize=(6,10))
axs[0].scatter(pnts_obs[:,0],pnts_obs[:,1],c=u_obs,s=50,
               vmin=-1.0,vmax=1.0,cmap='viridis',zorder=1)
p = axs[0].tripcolor(grid[:,0],grid[:,1],signal(grid),
                     vmin=-1.0,vmax=1.0,cmap='viridis',zorder=0)
axs[0].plot(bnd_vert[:,0],bnd_vert[:,1],'r-',lw=4)
axs[0].set_xlim((-4,4));axs[0].set_ylim((-4,4))
axs[0].set_aspect('equal')
axs[0].set_title(u'observed data and true signal')
plt.colorbar(p,ax=axs[0])
axs[1].scatter(pnts_obs[:,0],pnts_obs[:,1],c=soln,s=50,
               vmin=-1.0,vmax=1.0,cmap='viridis')
p = axs[1].tripcolor(grid[:,0],grid[:,1],signal(grid),
                     vmin=-1.0,vmax=1.0,cmap='viridis',zorder=0)
axs[1].plot(bnd_vert[:,0],bnd_vert[:,1],'r-',lw=4)
axs[1].set_xlim((-4,4));axs[1].set_ylim((-4,4))
axs[1].set_aspect('equal')
axs[1].set_title(r'filtered data and true signal ($\mathregular{\omega_c=%s}$)' % cutoff)
plt.colorbar(p,ax=axs[1])
plt.tight_layout()
plt.savefig('../figures/filter.c.png')
plt.show()
