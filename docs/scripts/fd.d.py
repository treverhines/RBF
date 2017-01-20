''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a 
uniform body force such as gravity. The domain has roller boundary 
condition on the bottom and sides. The top of domain has a free 
surface boundary condition. The top of the domain is generated 
randomly and is intended to simulate topography.
'''
import rbf
import numpy as np
from scipy.sparse import vstack,hstack,diags
from scipy.sparse.linalg import spsolve
from mayavi import mlab
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals,contains
from rbf.domain import topography
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement)

## User defined parameters
#####################################################################

def taper_function(x,center,radius,order=10):
  ''' 
  This function is effectively 1.0 within a sphere with radius 
  *radius* centered at *center*.  The function quickly drops to 0.0 
  outside of the sphere. This is used to ensure that the user-defined 
  topography tapers to zero at the domain edges.
  '''
  r = np.sqrt((x[:,0]-center[0])**2 + (x[:,1]-center[1])**2)
  return 1.0/(1.0 + (r/radius)**order)

def topo_func(x):
  ''' 
  This function generates a random topography at *x*. It takes an 
  (N,2) array of surface positions and returns an (N,) array of 
  elevations.  For real-world applications this should be a function 
  that interpolates a DEM.
  '''
  np.random.seed(3)
  gp = rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,0.25))
  gp += rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,0.5))
  gp += rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,1.0))
  out = gp.draw_sample(x)
  out *= taper_function(x,[0.0,0.0],1.0)
  return out

def density_func(x):
  ''' 
  This function describes the desired node density at *x*. It takes an 
  (N,3) array of positions and returns an (N,) array of normalized 
  node densities. This function is normalized such that all values are 
  between 0.0 and 1.0.
  '''
  z = x[:,2]
  out = np.zeros(x.shape[0])
  out[z > -0.5] = 1.0
  out[z <= -0.5] = 0.25
  return out

# generates the domain according to topo_func 
vert,smp = topography(topo_func,[-1.3,1.3],[-1.3,1.3],1.0,n=60)
# number of nodes 
N = 2000
# size of RBF-FD stencils
n = 35
# Lame parameters
lamb = 1.0
mu = 1.0
# z component of body force 
body_force = 1.0

## Build and solve for topographic stress
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
  
def find_orthogonals(n):
  ''' 
  Returns two arrays of normalized vector that are orthogonal to *n*.
  This is used to determine the directions along which zero traction 
  constraints will be imposed.
    
  Parameters
  ----------
  n : (N,2) float array
    
  Returns
  -------
  out1 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in *n*.

  out2 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in *n* and in *out1*
    
  '''
  out1 = np.empty_like(n,dtype=float)
  out1[:,0] = -np.sum(n,axis=1) + n[:,0]
  out1[:,1:] = n[:,[0]] 
  out1 /= np.linalg.norm(out1,axis=1)[:,None] # normalize length
  out2 = np.cross(n,out1) # find vectors that are orthogonal to *n* and *out1*
  out2 /= np.linalg.norm(out2,axis=1)[:,None] # normalize length
  return out1,out2

# generate nodes. Note that this may take a while
nodes,smpid = menodes(N,vert,smp,itr=50,rho=density_func)
# find which nodes are attached to each simplex
int_idx = np.nonzero(smpid == -1)[0].tolist()
roller_idx = np.nonzero((smpid >= 0) & (smpid <= 9))[0].tolist()
free_idx = np.nonzero(smpid > 9)[0].tolist()
# find normal vectors to each free surface node
simplex_normals = simplex_outward_normals(vert,smp)
free_normals = simplex_normals[smpid[free_idx]]
# find the normal vectors to each roller node
roller_normals = simplex_normals[smpid[roller_idx]]
# find two orthogonal vectors that are parallel to the surface at each 
# roller node. This is used to determine the directions along which 
# traction forces will be constrained. Note that any two orthogonal 
# vectors that are parallel to the surface would do.
roller_parallels1,roller_parallels2 = find_orthogonals(roller_normals)
# add ghost nodes next to free and roller nodes
dx = min_distance(nodes)
nodes = np.vstack((nodes,nodes[free_idx] + dx*free_normals))
nodes = np.vstack((nodes,nodes[roller_idx] + dx*roller_normals))
# build the "left hand side" matrices for body force constraints
A_body = elastic3d_body_force(nodes[int_idx+free_idx+roller_idx],nodes,lamb=lamb,mu=mu,n=n)
A_body_x,A_body_y,A_body_z = (hstack(i) for i in A_body)
# build the "right hand side" vectors for body force constraints
b_body_x = np.zeros_like(int_idx+free_idx+roller_idx)
b_body_y = np.zeros_like(int_idx+free_idx+roller_idx)
b_body_z = body_force*np.ones_like(int_idx+free_idx+roller_idx)
# build the "left hand side" matrices for free surface constraints
A_surf = elastic3d_surface_force(nodes[free_idx],free_normals,nodes,lamb=lamb,mu=mu,n=n)
A_surf_x,A_surf_y,A_surf_z = (hstack(i) for i in A_surf)
# build the "right hand side" vectors for free surface constraints
b_surf_x = np.zeros_like(free_idx)
b_surf_y = np.zeros_like(free_idx)
b_surf_z = np.zeros_like(free_idx)
# build the "left hand side" matrices for roller constraints
# constrain displacements in the surface normal direction
A_roller = elastic3d_displacement(nodes[roller_idx],nodes,lamb=lamb,mu=mu,n=1)
A_roller_x,A_roller_y,A_roller_z = (hstack(i) for i in A_roller)
normals_x = diags(roller_normals[:,0])
normals_y = diags(roller_normals[:,1])
normals_z = diags(roller_normals[:,2])
A_roller_n = (normals_x.dot(A_roller_x) + 
              normals_y.dot(A_roller_y) +
              normals_z.dot(A_roller_z))
# constrain surface traction in the surface parallel directions
A_roller = elastic3d_surface_force(nodes[roller_idx],roller_normals,nodes,lamb=lamb,mu=mu,n=n)
A_roller_x,A_roller_y,A_roller_z = (hstack(i) for i in A_roller)
parallels_x = diags(roller_parallels1[:,0])
parallels_y = diags(roller_parallels1[:,1])
parallels_z = diags(roller_parallels1[:,2])
A_roller_p1 = (parallels_x.dot(A_roller_x) + 
               parallels_y.dot(A_roller_y) +
               parallels_z.dot(A_roller_z))
parallels_x = diags(roller_parallels2[:,0])
parallels_y = diags(roller_parallels2[:,1])
parallels_z = diags(roller_parallels2[:,2])
A_roller_p2 = (parallels_x.dot(A_roller_x) + 
               parallels_y.dot(A_roller_y) +
               parallels_z.dot(A_roller_z))
# build the "right hand side" vectors for roller constraints
b_roller_n = np.zeros_like(roller_idx) # enforce zero normal displacement
b_roller_p1 = np.zeros_like(roller_idx) # enforce zero parallel traction
b_roller_p2 = np.zeros_like(roller_idx) # enforce zero parallel traction
# stack it all together and solve
A = vstack((A_body_x,A_body_y,A_body_z,
            A_surf_x,A_surf_y,A_surf_z,
            A_roller_n,A_roller_p1,A_roller_p2)).tocsr()
b = np.hstack((b_body_x,b_body_y,b_body_z,
               b_surf_x,b_surf_y,b_surf_z,
               b_roller_n,b_roller_p1,b_roller_p2))
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
# Calculate stress from Hooks law
s_xx = (2*mu + lamb)*e_xx + lamb*e_yy + lamb*e_zz
s_yy = lamb*e_xx + (2*mu + lamb)*e_yy + lamb*e_zz
s_zz = lamb*e_xx + lamb*e_yy + (2*mu + lamb)*e_zz
s_xy = 2*mu*e_xy
s_xz = 2*mu*e_xz
s_yz = 2*mu*e_yz

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
e_xx,e_yy,e_zz = e_xx[:N],e_yy[:N],e_zz[:N]
e_xy,e_xz,e_yz = e_xy[:N],e_xz[:N],e_yz[:N]
s_xx,s_yy,s_zz = s_xx[:N],s_yy[:N],s_zz[:N]
s_xy,s_xz,s_yz = s_xy[:N],s_xz[:N],s_yz[:N]
cmap = cm.viridis
# Plot this component of the stress tensor
stress = s_zz
stress_label = 'vertical stress'
fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
# turn second invariant into structured data
dat = make_scalar_field(nodes,stress,bnd_vert=vert,bnd_smp=smp)
# plot the top surface simplices
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2]+1e-2,smp[10:],opacity=1.0,colormap='gist_earth',vmin=-1.0,vmax=0.25)
# plot the bottom simplices
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[:2],color=(0.0,0.0,0.0),opacity=0.5)
p = mlab.pipeline.iso_surface(dat,opacity=0.5,contours=7)
mlab.quiver3d(nodes[:,0],nodes[:,1],nodes[:,2],u_x,u_y,u_z,mode='arrow',color=(0.2,0.2,0.2),scale_factor=1.0)
# set colormap to cmap
colors = cmap(np.linspace(0.0,1.0,256))*255
p.module_manager.scalar_lut_manager.lut.table = colors
# add colorbar
cbar = mlab.scalarbar(p,title=stress_label)
cbar.number_of_labels = 5
cbar.title_text_property.bold = False
cbar.title_text_property.italic = False
cbar.label_text_property.bold = False
cbar.label_text_property.italic = False
mlab.show()

