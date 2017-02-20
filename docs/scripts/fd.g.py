''' 
In this script we solve the 2-d wave equation over an L-shaped domain 
for in-plane motion with a RBF-FD scheme. The domain has free boundary 
conditions. Time integration is done with the fourth-order Runge-Kutta 
method.
'''
import numpy as np
from rbf.fd import weight_matrix
from rbf.nodes import menodes,neighbors
from rbf.geometry import contains,simplex_outward_normals
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force)
import matplotlib.pyplot as plt
from scipy.sparse import vstack,hstack
from scipy.integrate import ode
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve
import rbf.domain

# lame parameters
lamb,mu = 1.0,1.0
# define the problem domain
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
times = np.linspace(0.0,1.0,5) # output times
N = 50000 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
interior = np.nonzero(smpid==-1)[0].tolist() # identify boundary nodes
boundary = np.nonzero(smpid>=0)[0].tolist() # identify boundary nodes
Nint,Nbnd = len(boundary),len(boundary)
# calculate surface normal vector for each boundary node
normals = simplex_outward_normals(vert,smp)[smpid[boundary]]
# dx is the shortest distance between any two nodes
dx = np.min(neighbors(nodes,2)[1][:,1])
# add ghost nodes to greatly improve accuracy at the free surface
nodes = np.vstack((nodes,nodes[boundary] + 0.5*dx*normals))
ghost = range(N,N+len(boundary)) # ghost node indices
# create differentiation matrices for the interior and boundary nodes
D = elastic2d_body_force(nodes[interior+boundary],nodes,lamb=lamb,mu=mu,n=30)
D = vstack([hstack(d) for d in D])
dD = elastic2d_surface_force(nodes[boundary],normals,nodes,lamb=lamb,mu=mu,n=4)
dD_fuck = vstack([hstack(d) for d in dD]).tocsr()
dD1 = vstack([hstack([i[:,ghost] for i in j]) for j in dD]).tocsr()
dD2 = vstack([hstack([i[:,interior+boundary] for i in j]) for j in dD]).tocsr()
# create initial and boundary conditions
r = np.sqrt((nodes[interior+boundary,0] - 0.5)**2 + 
            (nodes[interior+boundary,1] - 0.5)**2)
u1_init = 1.0/(1 + (r/0.05)**4) # initial u in the interior
u2_init = 1.0/(1 + (r/0.05)**4) # initial u in the interior
du1dt_init = np.zeros(N) # initial velocity in the interior
du2dt_init = np.zeros(N) # initial velocity in the interior
# Make state vector containing the initial displacements and velocities
v = np.hstack([u1_init,u2_init,du1dt_init,du2dt_init])

def f(t,v):
  ''' 
  Function used for time integration. This calculates the time 
  derivative of the current state vector. 
  '''
  v = v.reshape((2,-1))
  u = np.empty((2,nodes.shape[0]))
  u[:,interior+boundary] = v[0].reshape((2,-1))
  u[:,ghost] = spsolve(dD1,-dD2.dot(v[0])).reshape((2,-1))
  u = u.ravel()
  print(np.sum(np.abs(dD_fuck.dot(u))))
  print(t)
  # return time derivative of the state vector
  return np.hstack([v[1],D.dot(u)])
              
# perform time integration with 'dopri5', which is Runge Kutta 
integrator = ode(f).set_integrator('dopri5',nsteps=5000)
integrator.set_initial_value(v,times[0])
soln = []
for t in times[1:]:
  # calculate state vector at time *t*
  v = integrator.integrate(t).reshape((2,2,-1))
  soln += [v[0]] # only save displacements

# plot the results
fig,axs = plt.subplots(2,2,figsize=(7,7))
for i,t in enumerate(times[1:]):
  ax = axs.ravel()[i]
  ax.plot(nodes[ghost,0],nodes[ghost,1],'k.')
  ax.plot(nodes[boundary,0],nodes[boundary,1],'b.')
  xg,yg = np.mgrid[0.0:2.0:200j,0:2.0:200j]
  points = np.array([xg.ravel(),yg.ravel()]).T
  # interpolate the solution onto a grid
  ug = griddata(nodes[interior+boundary],np.linalg.norm(soln[i],axis=0),(xg,yg),method='linear')
  # mask the points outside of the domain
  ug.ravel()[~contains(points,vert,smp)] = np.nan 
  # plot the boudary
  for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
  ax.imshow(ug,extent=(0.0,2.0,0.0,2.0),origin='lower',
            vmin=-0.2,vmax=0.2,cmap='seismic')
  ax.set_aspect('equal')
  ax.text(0.6,0.85,'time : %s\nnodes : %s' % (t,N),
          transform=ax.transAxes,fontsize=10)
  ax.tick_params(labelsize=10)
  ax.set_axis_bgcolor((0.8,0.8,0.8))
  ax.set_xlim(-0.1,2.1);ax.set_ylim(-0.1,2.1)
    
plt.tight_layout()    
plt.savefig('../figures/fd.g.png')
plt.show()
