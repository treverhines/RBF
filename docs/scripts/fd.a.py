''' 
In this script we solve the 2-d wave equation over an L-shaped domain 
with a RBF-FD scheme. The domain has fixed boundary conditions. See 
fd.f.py for an example with free boundary conditions. Time integration 
is done with the fourth-order Runge-Kutta method.
'''
import numpy as np
from rbf.fd import weight_matrix
from rbf.nodes import min_energy_nodes
from rbf.geometry import contains
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import griddata

# define the problem domain
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
times = np.linspace(0.0,2.0,5) # output times
N = 40000 # total number of nodes
nodes,idx,_ = min_energy_nodes(N,vert,smp) # generate nodes
# create differentiation matrices for the interior and boundary nodes
D = weight_matrix(nodes[idx['interior']],nodes,[(2,0),(0,2)],n=30)
# create initial and boundary conditions
r = np.sqrt((nodes[idx['interior'],0] - 0.5)**2 + 
            (nodes[idx['interior'],1] - 0.5)**2)
u_init = 1.0/(1 + (r/0.05)**4) # initial u in the interior
dudt_init = np.zeros(len(idx['interior'])) # initial velocity in the interior
u_bnd = np.zeros(len(idx['boundary'])) # boundary conditions
# Make state vector containing the initial displacements and velocities
v = np.hstack([u_init,dudt_init])

def f(t,v):
  ''' 
  Function used for time integration. This calculates the time 
  derivative of the current state vector. 
  '''
  v = v.reshape((2,-1))
  u = np.empty(N)
  u[idx['interior']],u[idx['boundary']] = v[0],u_bnd
  return np.hstack([v[1],D.dot(u)])
              
# perform time integration with 'dopri5', which is Runge Kutta 
integrator = ode(f).set_integrator('dopri5',nsteps=1000)
integrator.set_initial_value(v,times[0])
soln = []
for t in times[1:]:
  # calculate state vector at time *t*
  v = integrator.integrate(t).reshape((2,-1))
  soln += [v[0]] # only save displacements

# plot the results
fig,axs = plt.subplots(2,2,figsize=(7,7))
for i,t in enumerate(times[1:]):
  ax = axs.ravel()[i]
  xg,yg = np.mgrid[0.0:2.0:200j,0:2.0:200j]
  points = np.array([xg.ravel(),yg.ravel()]).T
  # interpolate the solution onto a grid
  u = np.empty(N)
  u[idx['interior']],u[idx['boundary']] = soln[i],u_bnd 
  ug = griddata(nodes,u,(xg,yg),method='linear')
  # mask the points outside of the domain
  ug.ravel()[~contains(points,vert,smp)] = np.nan 
  # plot the boudary
  for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
  ax.imshow(ug,extent=(0.0,2.0,0.0,2.0),origin='lower',vmin=-0.2,vmax=0.2,cmap='seismic')
  ax.set_aspect('equal')
  ax.text(0.6,0.85,'time : %s\nnodes : %s' % (t,N),transform=ax.transAxes,fontsize=10)
  ax.tick_params(labelsize=10)
  ax.set_xlim(-0.1,2.1)
  ax.set_ylim(-0.1,2.1)
    
plt.tight_layout()    
plt.savefig('../figures/fd.a.png')
plt.show()
