''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a three-dimensional elastic material subject to a 
uniform body force such as gravity. The elastic material has a fixed 
boundary condition on the bottom and the remaining sides have a free 
surface boundary condition. The top of the domain is generated 
randomly and is intended to simulate topography. 
'''
import rbf
import numpy as np
import scipy.sparse
from mayavi import mlab
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals
from rbf.domain import topography
np.random.seed(3)

#####################################################################
####################### USER PARAMETERS #############################
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
  This function generates a random topography at *x*. For real-world 
  applications this should be a function that interpolates a DEM.
  '''
  gp = rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,0.25))
  gp += rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,0.5))
  gp += rbf.gpr.PriorGaussianProcess(rbf.basis.ga,(0.0,0.01,1.0))
  out = gp.draw_sample(x)
  out *= taper_function(x,[0.0,0.0],1.0)
  return out

# generates the domain according to topo_func 
vert,smp = topography(topo_func,[-1.3,1.3],[-1.3,1.3],1.0,n=60)

## uncomment to view the domain
##fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
##mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,
##                     colormap='gist_earth',vmin=-1.0,vmax=0.25)
#mlab.show()                     

# number of nodes 
N = 10000
# size of RBF-FD stencils
n = 30
# lame parameters
lamb = 1.0
mu = 1.0


#####################################################################
#####################################################################
#####################################################################
def mindist(x):
  ''' 
  returns the shortest distance between any two nodes in x. This is 
  used to determine how far outside the boundary to place ghost nodes
  '''
  kd = cKDTree(x)
  dist,_ = kd.query(x,2)
  return np.min(dist[:,1])
  
# generate nodes. Note that this may take a while
nodes,smpid = menodes(N,vert,smp,itr=100)
# find which nodes at attached to each simplex
interior, = np.nonzero(smpid == -1) 
interior = list(interior)
fix_boundary, = np.nonzero((smpid == 0) | (smpid == 1))
fix_boundary = list(fix_boundary)
free_boundary, = np.nonzero(smpid > 1)
free_boundary = list(free_boundary)
# find normal vectors to each free surface node
simplex_normals = simplex_outward_normals(vert,smp)
normals = simplex_normals[smpid[free_boundary]]
# add ghost nodes next to free surface nodes
dx = mindist(nodes)
nodes = np.vstack((nodes,
                   nodes[free_boundary] + dx*normals))

## uncomment to view nodes
##fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
##mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,
##                     colormap='gist_earth',vmin=-1.0,vmax=0.25,opacity=0.75)
##mlab.points3d(nodes[:,0],nodes[:,1],nodes[:,2],color=(0.0,0.0,0.0),scale_factor=0.025)
##mlab.points3d(nodes[free_boundary,0],nodes[free_boundary,1],nodes[free_boundary,2],
##              color=(1.0,0.0,0.0),scale_factor=0.05)
##mlab.points3d(nodes[fix_boundary,0],nodes[fix_boundary,1],nodes[fix_boundary,2],
##              color=(0.0,0.0,1.0),scale_factor=0.05)
##mlab.show()                     

## Enforce the PDE on interior node AND the free surface nodes
#####################################################################
coeffs_xx = [lamb+2*mu,      mu,       mu]
diffs_xx =  [  (2,0,0), (0,2,0),  (0,0,2)]
coeffs_xy = [lamb+mu]
diffs_xy =  [(1,1,0)]
coeffs_xz = [lamb+mu]
diffs_xz =  [(1,0,1)]
coeffs_yx = [lamb+mu]
diffs_yx =  [(1,1,0)]
coeffs_yy = [     mu, lamb+2*mu,      mu]
diffs_yy =  [(2,0,0),   (0,2,0), (0,0,2)]
coeffs_yz = [lamb+mu]
diffs_yz =  [(0,1,1)]
coeffs_zx = [lamb+mu]
diffs_zx =  [(1,0,1)]
coeffs_zy = [lamb+mu]
diffs_zy =  [(0,1,1)]
coeffs_zz = [     mu,      mu, lamb+2*mu]
diffs_zz =  [(2,0,0), (0,2,0),   (0,0,2)]
D_xx = weight_matrix(nodes[interior+free_boundary],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
D_xy = weight_matrix(nodes[interior+free_boundary],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
D_xz = weight_matrix(nodes[interior+free_boundary],nodes,diffs_xz,coeffs=coeffs_xz,n=n)
D_yx = weight_matrix(nodes[interior+free_boundary],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
D_yy = weight_matrix(nodes[interior+free_boundary],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
D_yz = weight_matrix(nodes[interior+free_boundary],nodes,diffs_yz,coeffs=coeffs_yz,n=n)
D_zx = weight_matrix(nodes[interior+free_boundary],nodes,diffs_zx,coeffs=coeffs_zx,n=n)
D_zy = weight_matrix(nodes[interior+free_boundary],nodes,diffs_zy,coeffs=coeffs_zy,n=n)
D_zz = weight_matrix(nodes[interior+free_boundary],nodes,diffs_zz,coeffs=coeffs_zz,n=n)
D_x = scipy.sparse.hstack((D_xx,D_xy,D_xz))
D_y = scipy.sparse.hstack((D_yx,D_yy,D_yz))
D_z = scipy.sparse.hstack((D_zx,D_zy,D_zz))
D = scipy.sparse.vstack((D_x,D_y,D_z))

## Enforce fixed boundary conditions
#####################################################################
dD_fix_xx = weight_matrix(nodes[fix_boundary],nodes,(0,0,0),n=1)
dD_fix_yy = weight_matrix(nodes[fix_boundary],nodes,(0,0,0),n=1)
dD_fix_zz = weight_matrix(nodes[fix_boundary],nodes,(0,0,0),n=1)
dD_fix = scipy.sparse.block_diag((dD_fix_xx,dD_fix_yy,dD_fix_zz))

## Enforce free surface boundary conditions
#####################################################################
coeffs_xx = [normals[:,0]*(lamb+2*mu), normals[:,1]*mu, normals[:,2]*mu]
diffs_xx =  [                 (1,0,0),         (0,1,0),         (0,0,1)]
coeffs_xy = [normals[:,0]*lamb, normals[:,1]*mu]
diffs_xy =  [          (0,1,0),         (1,0,0)]
coeffs_xz = [normals[:,0]*lamb, normals[:,2]*mu]
diffs_xz =  [          (0,0,1),         (1,0,0)]
coeffs_yx = [normals[:,0]*mu, normals[:,1]*lamb]
diffs_yx =  [        (0,1,0),           (1,0,0)]
coeffs_yy = [normals[:,0]*mu, normals[:,1]*(lamb+2*mu), normals[:,2]*mu]
diffs_yy =  [        (1,0,0),                  (0,1,0),         (0,0,1)]
coeffs_yz = [normals[:,1]*lamb, normals[:,2]*mu]
diffs_yz =  [          (0,0,1),         (0,1,0)]
coeffs_zx = [normals[:,0]*mu, normals[:,2]*lamb]
diffs_zx =  [        (0,0,1),           (1,0,0)]
coeffs_zy = [normals[:,1]*mu, normals[:,2]*lamb]
diffs_zy =  [        (0,0,1),           (0,1,0)]
coeffs_zz = [normals[:,0]*mu, normals[:,1]*mu, normals[:,2]*(lamb+2*mu)]
diffs_zz =  [        (1,0,0),         (0,1,0),                  (0,0,1)]
dD_free_xx = weight_matrix(nodes[free_boundary],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
dD_free_xy = weight_matrix(nodes[free_boundary],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
dD_free_xz = weight_matrix(nodes[free_boundary],nodes,diffs_xz,coeffs=coeffs_xz,n=n)
dD_free_yx = weight_matrix(nodes[free_boundary],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
dD_free_yy = weight_matrix(nodes[free_boundary],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
dD_free_yz = weight_matrix(nodes[free_boundary],nodes,diffs_yz,coeffs=coeffs_yz,n=n)
dD_free_zx = weight_matrix(nodes[free_boundary],nodes,diffs_zx,coeffs=coeffs_zx,n=n)
dD_free_zy = weight_matrix(nodes[free_boundary],nodes,diffs_zy,coeffs=coeffs_zy,n=n)
dD_free_zz = weight_matrix(nodes[free_boundary],nodes,diffs_zz,coeffs=coeffs_zz,n=n)
dD_free_x = scipy.sparse.hstack((dD_free_xx,dD_free_xy,dD_free_xz))
dD_free_y = scipy.sparse.hstack((dD_free_yx,dD_free_yy,dD_free_yz))
dD_free_z = scipy.sparse.hstack((dD_free_zx,dD_free_zy,dD_free_zz))
dD_free = scipy.sparse.vstack((dD_free_x,dD_free_y,dD_free_z))

## Create the "right hand side" vector components
#####################################################################
# create body force components
f_x = np.zeros(len(interior+free_boundary))
f_y = np.zeros(len(interior+free_boundary)) 
f_z = np.ones(len(interior+free_boundary)) # THIS IS WHERE GRAVITY IS ADDED
f = np.hstack((f_x,f_y,f_z))
# create fixed boundary condition components
fix_x = np.zeros(len(fix_boundary))
fix_y = np.zeros(len(fix_boundary))
fix_z = np.zeros(len(fix_boundary))
fix = np.hstack((fix_x,fix_y,fix_z))
# create traction vectors for free surface boundary conditions
free_x = np.zeros(len(free_boundary))
free_y = np.zeros(len(free_boundary))
free_z = np.zeros(len(free_boundary))
free = np.hstack((free_x,free_y,free_z))

## Combine and solve
#####################################################################
G = scipy.sparse.vstack((D,dD_fix,dD_free))
G = G.tocsc()
G.eliminate_zeros()
d = np.hstack((f,fix,free))
u = scipy.sparse.linalg.spsolve(G,d)
u = np.reshape(u,(3,-1))
u_x,u_y,u_z = u

## Calculate strain from displacements
#####################################################################
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
g = len(free_boundary)
# remove ghost nodes
nodes = nodes[:-g]
u_x = u_x[:-g]
u_y = u_y[:-g]
u_z = u_z[:-g]
I2 = I2[:-g]

def make_scalar_field(nodes,vals,step=200j,
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
  f = griddata(nodes, vals, (x,y,z),method='linear')
  out = mlab.pipeline.scalar_field(x,y,z,f)
  return out

# set strain invariant colormap
cmap = cm.viridis
# initiate figure
fig = mlab.figure(bgcolor=(0.9,0.9,0.9),fgcolor=(0.0,0.0,0.0),size=(600, 600))
# turn second invariant into structured data
dat = make_scalar_field(nodes,I2,zmax=0.0)
# plot the top surface simplices
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2]+0.01,smp[10:],
                     opacity=1.0,colormap='gist_earth',vmin=-1.0,vmax=0.25)
# plot the bottom simplices
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[:2],
                     color=(0.0,0.0,0.0),opacity=0.5)
# plot decimated displacement vectors
mlab.quiver3d(nodes[:,0],nodes[:,1],nodes[:,2],
              u_x,u_y,u_z,mode='arrow',color=(0.1,0.1,0.1),
              mask_points=5,scale_factor=1.0)
# make cross section for second invariant
p = mlab.pipeline.scalar_cut_plane(dat,vmin=0.0,vmax=0.25,
                                   plane_orientation='y_axes')
# set colormap to cmap
colors = cmap(np.linspace(0.0,1.0,256))*255
p.module_manager.scalar_lut_manager.lut.table = colors
# add colorbar
cbar = mlab.scalarbar(p,title='second strain invariant')
#cbar.lut.scale = 'log10'
# number of ticks on the colorbar
cbar.number_of_labels = 5
# change colorbar font
cbar.title_text_property.bold = False
cbar.title_text_property.italic = False
cbar.label_text_property.bold = False
cbar.label_text_property.italic = False
# orient the camera position
eng = mlab.get_engine()
scene = eng.scenes[0]
scene.scene.camera.pitch(-3.5)
scene.scene.camera.orthogonalize_view_up()
mlab.draw()
mlab.show()

