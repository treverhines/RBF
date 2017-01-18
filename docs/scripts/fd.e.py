''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a point 
surface force. The numerical solution is compared to the analytical 
solution for Boussinesq's problem.
'''
import rbf
import numpy as np
from scipy.sparse import vstack,hstack
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals
from rbf.domain import circle
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force,
                         elastic2d_displacement)
import time
np.random.seed(3)
def delta(i,j):
  if i == j:
    return 1.0
  else:
    return 0.0

def point_force(x,lamb=1.0,mu=1.0):
  ''' 
  Solves for displacements resulting from a unit point force normal to 
  the surface of an elastic halfspace. The point force is applied at 
  (x,y,z) = (0.0,0.0,0.0) and the halfspace is z<0.0.

  '''
  # the solution I am using assumes that z is positive in the 
  # halfspace.
  x = np.copy(x)
  x[:,2] *= -1.0
  nu = lamb/(2*(lamb+mu))
  r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
  out = np.zeros(np.shape(x))
  for i in range(3):
    out[:,i] = (1.0/(4*np.pi*mu)*
                (x[:,2]*x[:,i]/r**3 +
                (3 - 4*nu)*delta(i,2)/r -
                (1 - 2*nu)*(delta(i,2) + x[:,i]/r)/(r + x[:,2])))

  # change displacements z direction so that down is positive 
  out[:,2] *= -1.0
  return out
                                                                                    

#####################################################################
####################### USER PARAMETERS #############################
#####################################################################
def initial_displacement(x):
  out = np.zeros((2,x.shape[0]))
  sigma = 0.1
  c = 1.0/np.sqrt((2*np.pi*sigma**2)**2)
  r = np.sqrt(x[:,0]**2 + x[:,1]**2)
  out[0,:] = c*np.exp(-0.5*(r/sigma)**2)
  return out

def initial_velocity(x):
  out = np.zeros((2,x.shape[0]))
  return out

# center on the origin
vert,smp = rbf.domain.circle(4)
# number of nodes 
N = 1000
# size of RBF-FD stencils
n = 20
# time step size
dt = 0.002
# time steps
T = 3000
# Lame parameters
lamb = 1.0
mu = 1.0
#####################################################################
#####################################################################
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
nodes,smpid = menodes(N,vert,smp,itr=200)
# find which nodes at attached to each simplex
int_idx = np.nonzero(smpid == -1)[0].tolist()
bnd_idx = np.nonzero(smpid >= 0)[0].tolist()
# find normal vectors to each free surface node
simplex_normals = simplex_outward_normals(vert,smp)
normals = simplex_normals[smpid[bnd_idx]]
# add ghost nodes next to free surface nodes
dx = min_distance(nodes)
nodes = np.vstack((nodes,nodes[bnd_idx] + dx*normals))
gst_idx = range(N,N+len(bnd_idx))

# Build "left hand side" matrix
D  = elastic2d_body_force(nodes[:N],nodes,lamb=lamb,mu=mu,n=n)
D  = vstack(hstack(i) for i in D).tocsr()
dD = elastic2d_surface_force(nodes[bnd_idx],nodes,normals,lamb=lamb,mu=mu,n=n)
dD = vstack(hstack(i) for i in dD).tocsr()
idx = np.arange(nodes.size).reshape(nodes.T.shape)[:,N:].flatten()
print(idx.shape)

axs = []
u_curr = initial_displacement(nodes)
u_curr[:,N:] = 0.0
u_prev = u_curr - dt*initial_velocity(nodes)
u_prev[:,N:] = 0.0
for i in range(T):
  u_next = np.zeros_like(u_curr)  
  force = D.dot(u_curr.flatten()).reshape((2,-1))
  damping = 0.0*(u_curr[:,:N] - u_prev[:,:N])/dt
  acc = dt**2*(force - damping)
  u_next[:,:N] = acc + 2*u_curr[:,:N] - u_prev[:,:N]
  d = -dD.dot(u_next.flatten()) 
  u_next[:,N:] = spsolve(dD[:,idx],d).reshape((2,-1))  
  u_prev = np.copy(u_curr)
  u_curr = np.copy(u_next)
  if i%500 == 0:
    fig,ax = plt.subplots()
    ax.set_title('%s' % i)
    for s in smp:
      ax.plot(vert[s,0],vert[s,1],'-')

    u_x,u_y = u_curr
    u_x = u_x[:N]
    u_y = u_y[:N]
    nodes = nodes[:N]
    ax.plot(nodes[:,0],nodes[:,1],'k.')
    ax.quiver(nodes[:,0],nodes[:,1],u_x,u_y,scale=50.0)
    ax.set_aspect('equal')
    axs += [ax]
    
plt.show()
quit()


#G += elastic3d_displacement(nodes[fix_idx],nodes,lamb=lamb,mu=mu,n=1)
G.eliminate_zeros()
# Build "right hand side" vector
body_force_x = np.zeros(len(int_idx+free_idx))
body_force_y = np.zeros(len(int_idx+free_idx)) 
body_force_z = np.zeros(len(int_idx+free_idx)) # THIS IS WHERE GRAVITY IS ADDED
surf_force_x = np.zeros(len(free_idx))
surf_force_y = np.zeros(len(free_idx))
surf_force_z = surface_force(nodes[free_idx])
disp_x = np.zeros(len(fix_idx))
disp_y = np.zeros(len(fix_idx))
disp_z = np.zeros(len(fix_idx))
body_force = np.hstack((body_force_x,body_force_y,body_force_z))
surf_force = np.hstack((surf_force_x,surf_force_y,surf_force_z))
disp = np.hstack((disp_x,disp_y,disp_z))
d = np.hstack((body_force,surf_force,disp))
# Combine and solve
G = G.tocsc()
u = spsolve(G,d)
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
# I define the second strain invariant as the sum of squared 
# components of the strain tensor. Others may define it differently
I2 = np.sqrt(e_xx**2 + e_yy**2 + e_zz**2 + 
             2*e_xy**2 + 2*e_xz**2 + 2*e_yz**2)

## Plot the results
#####################################################################
g = len(free_idx)
# remove ghost nodes
nodes = nodes[:-g]
u_x,u_y,u_z = u_x[:-g],u_y[:-g],u_z[:-g]
I2 = I2[:-g]

def make_scalar_field(nodes,vals,step=100j,
                      xmin=None,xmax=None,
                      ymin=None,ymax=None,
                      zmin=None,zmax=None):
  ''' 
  Returns a structured data object used for plotting with Mayavi
  '''
  if xmin is None:
    xmin = np.min(nodes[:,0])
  if xmax is None:
    xmax = np.max(nodes[:,0])
  if ymin is None:
    ymin = np.min(nodes[:,1])
  if ymax is None:
    ymax = np.max(nodes[:,1])
  if zmin is None:
    zmin = np.min(nodes[:,2])
  if zmax is None:
    zmax = np.max(nodes[:,2])

  x,y,z = np.mgrid[xmin:xmax:step,ymin:ymax:step,zmin:zmax:step]
  f = griddata(nodes, vals, (x,y,z),method='nearest')
  out = mlab.pipeline.scalar_field(x,y,z,f)
  return out

# set strain invariant colormap
cmap = cm.viridis
# initiate figure
fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
# turn second invariant into structured data
dat = make_scalar_field(nodes,I2,zmax=0.0)
# plot the top surface simplices
#mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2]+0.01,smp[10:],opacity=1.0,colormap='gist_earth',vmin=-1.0,vmax=0.25)
# plot the bottom simplices
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[:2],color=(0.0,0.0,0.0),opacity=0.5)
# plot decimated displacement vectors
mlab.quiver3d(nodes[:,0],nodes[:,1],nodes[:,2],u_x,u_y,u_z,mode='arrow',color=(0.1,0.1,0.1),scale_factor=0.1)
# make cross section for second invariant
p = mlab.pipeline.scalar_cut_plane(dat,vmin=0.0,vmax=3.0,plane_orientation='y_axes')
# set colormap to cmap
colors = cmap(np.linspace(0.0,1.0,256))*255
p.module_manager.scalar_lut_manager.lut.table = colors
# add colorbar
cbar = mlab.scalarbar(p,title='second strain invariant')
cbar.lut.scale = 'log10'
cbar.number_of_labels = 5
cbar.title_text_property.bold = False
cbar.title_text_property.italic = False
cbar.label_text_property.bold = False
cbar.label_text_property.italic = False
mlab.show()

