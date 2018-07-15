''' 
In this example we generate synthetic scattered data with added noise
and then fit it with a smoothed RBF interpolant. The interpolant in
this example is equivalent to a thin plate spline.
'''
import numpy as np
from rbf.interpolate import RBFInterpolant
import rbf.basis
import matplotlib.pyplot as plt
np.random.seed(1)

basis = rbf.basis.phs2
order = 1

x_obs = np.random.random((100,2)) # observation points
u_obs = np.sin(2*np.pi*x_obs[:,0])*np.cos(2*np.pi*x_obs[:,1]) # signal
u_obs += np.random.normal(0.0,0.2,100) # add noise to signal
I = RBFInterpolant(x_obs,u_obs,sigma=0.1,basis=basis,order=order)
vals = np.linspace(0,1,200)
x_itp = np.reshape(np.meshgrid(vals,vals),(2,200*200)).T # interp points
u_itp = I(x_itp) # evaluate the interpolant
# plot the results
plt.tripcolor(x_itp[:,0],x_itp[:,1],u_itp,vmin=-1.1,vmax=1.1,cmap='viridis')
plt.scatter(x_obs[:,0],x_obs[:,1],s=100,c=u_obs,vmin=-1.1,vmax=1.1,
            cmap='viridis',edgecolor='k')
plt.xlim((0.05,0.95))
plt.ylim((0.05,0.95))
plt.colorbar()
plt.tight_layout()
plt.savefig('../figures/interpolate.a.png')
plt.show()

