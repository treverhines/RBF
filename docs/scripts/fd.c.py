''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a 
uniform body force such as gravity. The elastic material has a fixed 
boundary condition on one side and the remaining sides have a free 
surface boundary condition.  This script also demonstrates using ghost 
nodes which, for all intents and purposes, are necessary when dealing 
with Neumann boundary conditions.
'''
import numpy as np
from mayavi import mlab
from matplotlib import cm
from scipy.sparse import vstack,hstack
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement) 

## User defined parameters
#####################################################################
# define the vertices of the problem domain. Note that the first two
# simplices will be fixed, and the others will be free
vert = np.array([[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],
                 [0.0,1.0,1.0],[2.0,0.0,0.0],[2.0,0.0,1.0],
                 [2.0,1.0,0.0],[2.0,1.0,1.0]])
smp = np.array([[0,1,3],[0,2,3],[0,1,4],[1,5,4],
                [0,2,6],[0,4,6],[1,7,5],[1,3,7],
                [4,5,7],[4,6,7],[2,3,7],[2,6,7]])
# number of nodes 
N = 1500
# size of RBF-FD stencils
n = 35
# lame parameters
lamb = 1.0
mu = 1.0
# z component of body force
body_force = 1.0

## Build and solve for displacements and strain
#####################################################################

def min_distance(x):
  ''' 
  Returns the shortest distance between any two nodes in *x*. This is 
  used to determine how far outside the boundary to place ghost nodes.

  Parameters
  ----------
  x : (N,D) array
    node positions
  
  Returns
  -------  
  out : float  
    shortest distance between any two nodes in *x*
    
  '''
  kd = cKDTree(x)
  dist,_ = kd.query(x,2)
  out = np.min(dist[:,1])
  return out
  
# generate nodes. Note that this may take a while
nodes,smpid = menodes(N,vert,smp)
# find which nodes are attached to each simplex
int_idx = np.nonzero(smpid == -1)[0].tolist()
fix_idx = np.nonzero((smpid == 0) | (smpid == 1))[0].tolist()
free_idx = np.nonzero(smpid > 1)[0].tolist()
# find normal vectors to each free surface node
simplex_normals = simplex_outward_normals(vert,smp)
normals = simplex_normals[smpid[free_idx]]
# add ghost nodes next to free surface nodes
dx = min_distance(nodes)
nodes = np.vstack((nodes,nodes[free_idx] + dx*normals))

# The "left hand side" matrices are built with the convenience 
# functions from *rbf.fdbuild*. Read the documentation for these 
# functions to better understand this step.
A = elastic3d_body_force(nodes[int_idx+free_idx],nodes,lamb=lamb,mu=mu,n=n)
A += elastic3d_surface_force(nodes[free_idx],normals,nodes,lamb=lamb,mu=mu,n=n)
A += elastic3d_displacement(nodes[fix_idx],nodes,lamb=lamb,mu=mu,n=1)
A = vstack(hstack(i) for i in A).tocsr()
# Create the "right hand side" vector components for body forces
f_x = np.zeros(len(int_idx+free_idx))
f_y = np.zeros(len(int_idx+free_idx)) 
f_z = body_force*np.ones(len(int_idx+free_idx)) # THIS IS WHERE GRAVITY IS ADDED
f = np.hstack((f_x,f_y,f_z))
# Create the "right hand side" vector components for surface tractions 
# constraints
fix_x = np.zeros(len(fix_idx))
fix_y = np.zeros(len(fix_idx))
fix_z = np.zeros(len(fix_idx))
fix = np.hstack((fix_x,fix_y,fix_z))
# Create the "right hand side" vector components for displacement 
# constraints
free_x = np.zeros(len(free_idx))
free_y = np.zeros(len(free_idx))
free_z = np.zeros(len(free_idx))
free = np.hstack((free_x,free_y,free_z))
# combine to form the full "right hand side" vector
b = np.hstack((f,fix,free))
# solve
u = spsolve(A,b,permc_spec='MMD_ATA')
u = np.reshape(u,(3,-1))
u_x,u_y,u_z = u
# Calculate strain from displacements
D_x = weight_matrix(nodes,nodes,(1,0,0),n=n)
D_y = weight_matrix(nodes,nodes,(0,1,0),n=n)
D_z = weight_matrix(nodes,nodes,(0,0,1),n=n)
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_zz = D_z.dot(u_z)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
e_xz = 0.5*(D_z.dot(u_x) + D_x.dot(u_z))
e_yz = 0.5*(D_z.dot(u_y) + D_y.dot(u_z))
I2 = np.sqrt(e_xx**2 + e_yy**2 + e_zz**2 + 
             2*e_xy**2 + 2*e_xz**2 + 2*e_yz**2)

## Plot the results
#####################################################################

def make_scalar_field(nodes,vals,step=100j,
                      bnd_vert=None,bnd_smp=None):
  ''' 
  Returns a structured data object used for plotting scalar fields 
  with Mayavi
  '''
  xmin = np.min(nodes[:,0])
  xmax = np.max(nodes[:,0])
  ymin = np.min(nodes[:,1])
  ymax = np.max(nodes[:,1])
  zmin = np.min(nodes[:,2])
  zmax = np.max(nodes[:,2])
  x,y,z = np.mgrid[xmin:xmax:step,ymin:ymax:step,zmin:zmax:step]
  f = griddata(nodes, vals, (x,y,z),method='linear')
  # mask all points that are outside of the domain
  grid_points_flat = np.array([x,y,z]).reshape((3,-1)).T
  if (bnd_smp is not None) & (bnd_vert is not None):
    is_outside = ~contains(grid_points_flat,bnd_vert,bnd_smp)
    is_outside = is_outside.reshape(x.shape)
    f[is_outside] = np.nan

  out = mlab.pipeline.scalar_field(x,y,z,f)
  return out

# remove ghost nodes
nodes = nodes[:N]
u_x,u_y,u_z = u_x[:N],u_y[:N],u_z[:N]
I2 = I2[:N]
cmap = cm.viridis
fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
# plot the domain
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[2:],color=(0.0,0.0,0.0),opacity=0.05)
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[:2],color=(0.0,0.0,0.0),opacity=0.5)
# plot displacement vectors
mlab.quiver3d(nodes[:,0],nodes[:,1],nodes[:,2],u_x,u_y,u_z,mode='arrow',color=(0.2,0.2,0.2),scale_factor=0.01)
# plot slice of second strain invariant
dat = make_scalar_field(nodes,I2)
p = mlab.pipeline.scalar_cut_plane(dat,vmin=0.1,vmax=3.2,plane_orientation='y_axes')
# add colorbar and set colormap to viridis
colors = cmap(np.linspace(0.0,1.0,256))*255
p.module_manager.scalar_lut_manager.lut.table = colors
cbar = mlab.scalarbar(p,title='second strain invariant')
cbar.lut.scale = 'log10'
cbar.number_of_labels = 5
cbar.title_text_property.bold = False
cbar.title_text_property.italic = False
cbar.label_text_property.bold = False
cbar.label_text_property.italic = False
mlab.show()

