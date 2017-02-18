''' 
In this script we solve the 2-d wave equation over an L-shaped domain 
with a RBF-FD scheme. Time integration is done with the fourth-order 
Runga-Kutta method. 
'''
import numpy as np
from rbf.fd import weight_matrix
from rbf.nodes import menodes
from rbf.geometry import contains,simplex_outward_normals
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import time

def min_distance(x):
  ''' 
  Returns the shortest distance between any two nodes in *x*. This is 
  used to determine how far outside the boundary to place ghost nodes.
  '''
  kd = cKDTree(x)
  dist,_ = kd.query(x,2)
  out = np.min(dist[:,1])
  return out
                
# define the problem domain
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
times = np.linspace(0.0,1.0,5) # output times
N = 10000 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
boundary, = (smpid>=0).nonzero() # identify boundary nodes
boundary = list(boundary)
# calculate surface normal vector for each boundary node
simplex_normals = simplex_outward_normals(vert,smp)
normals = simplex_normals[smpid[boundary]]
# add ghost nodes to greatly improve accuracy at the free surface
dx = 0.5*min_distance(nodes)
nodes_ext = np.vstack((nodes,nodes[boundary]+dx*normals))
# create differentiation matrices for the interior and boundary nodes
D = weight_matrix(nodes,nodes_ext,[(2,0),(0,2)],n=30)
dD = weight_matrix(nodes[boundary],nodes_ext,[(1,0),(0,1)],coeffs=normals.T,n=30)
# create initial and boundary conditions
r = np.sqrt((nodes[:,0] - 0.5)**2 + (nodes[:,1] - 0.5)**2)
u_init = 1.0/(1 + (r/0.05)**4) # initial u in the interior
dudt_init = np.zeros(N) # initial velocity in the interior
u_bnd = np.zeros(len(boundary)) # boundary conditions

def f(t,v):
  '''calculates the derivative of the current state''' 
  u = np.empty(len(nodes_ext))
  u[:N] = v[:N]
  u[N:] = spsolve(dD[:,N:],u_bnd - dD[:,:N].dot(v[:N]))
  return np.hstack([v[N:],D.dot(u)])
              
# state vector containing displacements and velocities
v = np.hstack([u_init,dudt_init])
integrator = ode(f).set_integrator('dopri5',nsteps=100000)
integrator.set_initial_value(v,times[0])
soln = []
a = time.time()
for t in times[1:]:
  soln += [integrator.integrate(t)[:N]]

print(time.time() - a)
# plot the results
fig,axs = plt.subplots(2,2,figsize=(7,7))
for i,t in enumerate(times[1:]):
  ax = axs.ravel()[i]
  xg,yg = np.mgrid[0.0:2.0:400j,0:2.0:400j]
  points = np.array([xg.ravel(),yg.ravel()]).T
  # combine the interior solution with the boundary conditions  
  u = soln[i]
  # interpolate the solution onto a grid
  ug = griddata(nodes,u,(xg,yg),method='linear')
  # mask the points outside of the domain
  ug.ravel()[~contains(points,vert,smp)] = np.nan 
  # plot the boudary
  for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
  # make contour plot
  ax.contourf(xg,yg,ug,cmap='viridis',vmin=-0.2,vmax=0.2,
              levels=np.linspace(-0.25,0.25,100))
  ax.set_aspect('equal')
  ax.text(0.6,0.85,'time : %s\nnodes : %s' % (t,N),
          transform=ax.transAxes,fontsize=10)
  ax.tick_params(labelsize=10)
  ax.set_xlim(-0.1,2.1)
  ax.set_ylim(-0.1,2.1)
    
plt.tight_layout()    
plt.show()

