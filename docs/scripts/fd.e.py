''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a point 
surface force. The numerical solution is compared to the analytical 
solution for Boussinesq's problem.
'''
import numpy as np
from scipy.sparse import vstack,hstack
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force,
                         elastic2d_displacement)

####################### USER PARAMETERS #############################
scale = 3.0
def node_density(x):
  c = np.array([0.0,1.0])
  r = (x[:,0] - c[0])**2 + (x[:,1] - c[1])**2
  out = 0.5/(1.0 + (r/1.0)**2)
  out += 0.5/(1.0 + (r/(2*scale))**2)
  return out

vert_bot = np.array([[scale,-scale],
                     [-scale,-scale]])
vert_top_x = np.linspace(-scale,scale,298)
vert_top_y = 1.0/(1 + vert_top_x**2) # define topography
vert_top = np.array([vert_top_x,vert_top_y]).T
vert = np.vstack((vert_bot,vert_top))
smp = np.array([np.arange(300),np.roll(np.arange(300),-1)]).T
# number of nodes 
N = 2000
# size of RBF-FD stencils
n = 20
# Lame parameters
lamb = 1.0
mu = 1.0
#####################################################################
def min_distance(x):
  ''' 
  returns the shortest distance between any two nodes in x. This is 
  used to determine how far outside the boundary to place ghost nodes
  '''
  kd = cKDTree(x)
  dist,_ = kd.query(x,2)
  return np.min(dist[:,1])
  
# generate nodes. Note that this may take a while
print('starting')
nodes,smpid = menodes(N,vert,smp,rho=node_density)
print('done')
# find which nodes at attached to each simplex
fix_idx = np.nonzero(smpid == 0)[0].tolist()
free_idx = np.nonzero(smpid > 0)[0].tolist()
#fix_idx = np.nonzero((smpid == 0) |
#                     (smpid == 1) |
#                     (smpid == 299))[0].tolist()
#free_idx = np.nonzero(~((smpid == -1) |
#                        (smpid == 0)  |
#                        (smpid == 1)  |
#                        (smpid == 299)))[0].tolist()
int_idx = np.nonzero(smpid == -1)[0].tolist()
# find normal vectors to each free surface node
simplex_normals = simplex_outward_normals(vert,smp)
normals = simplex_normals[smpid[free_idx]]
# add ghost nodes next to free surface nodes
dx = min_distance(nodes)
nodes = np.vstack((nodes,nodes[free_idx] + dx*normals))
# Build "left hand side" matrix
D  = elastic2d_body_force(nodes[int_idx+free_idx],nodes,lamb=lamb,mu=mu,n=n)
D += elastic2d_surface_force(nodes[free_idx],normals,nodes,lamb=lamb,mu=mu,n=n)
D += elastic2d_displacement(nodes[fix_idx],nodes,lamb=lamb,mu=mu,n=n)
D = vstack(hstack(i) for i in D).tocsr()
# set "right hand side" vector
body_x = np.zeros(len(int_idx)+len(free_idx))
body_y = np.ones(len(int_idx)+len(free_idx))
free_x = np.zeros(len(free_idx))
free_y = np.zeros(len(free_idx))
fix_x = np.zeros(len(fix_idx))
fix_y = np.zeros(len(fix_idx))
d = np.hstack((body_x,body_y,free_x,free_y,fix_x,fix_y))
# solve
u = spsolve(D,d)
u = np.reshape(u,(2,-1))
u_x,u_y = u
# Calculate strain from displacements
D_x = weight_matrix(nodes,nodes,(1,0),n=n)
D_y = weight_matrix(nodes,nodes,(0,1),n=n)
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
# I define the second strain invariant as the sum of squared 
# components of the strain tensor. Others may define it differently
I2 = np.sqrt(e_xx**2 + e_yy**2 + 2*e_xy**2)
# toss out ghosts
nodes = nodes[:N]
u_x = u_x[:N]
u_y = u_y[:N]
e_xx = e_xy[:N]
e_yy = e_yy[:N]
e_xy = e_xy[:N]
I2 = I2[:N]

fig,ax = plt.subplots()
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-')


from numpy import linspace, meshgrid
from matplotlib.mlab import griddata

def grid(x, y, z, resX=1000, resY=1000):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi,interp='linear')
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

X, Y, Z = grid(nodes[:,0],nodes[:,1],2*e_xy)
ax.contour(X, Y, Z,np.arange(-0.2,0.2,0.04),colors='k',corner_mask=True)
p = ax.tripcolor(nodes[:,0],nodes[:,1],2*e_xy,cmap='seismic',vmin=-0.2,vmax=0.2)
fig.colorbar(p)
ax.quiver(nodes[:,0],nodes[:,1],u_x,u_y,scale=20.0,zorder=1)
#ax.plot(nodes[:,0],nodes[:,1],'k.')
#ax.plot(nodes[fix_idx,0],nodes[fix_idx,1],'bo')
#ax.plot(nodes[free_idx,0],nodes[free_idx,1],'ro')
ax.set_xlim((0,3))
ax.set_ylim((-2,1))
ax.set_aspect('equal')
plt.show()
quit()


