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
import scipy.sparse
from mayavi import mlab
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals

#####################################################################
####################### USER PARAMETERS #############################
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
N = 2000
# size of RBF-FD stencils
n = 20
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
  
# generate nodes. Read the documentation for menodes to tune it and 
# allow for variable density nodes
nodes,smpid = menodes(N,vert,smp)
# find the indices for interior nodes, fixed boundary nodes, and free 
# boundary nodes. This is done by looking at the array *smpid* which 
# tells us the simplex that each boundary node is attached to.
interior, = np.nonzero(smpid == -1) 
interior = list(interior)
# fix the bottom nodes and keep other boundaries free
fix_boundary, = np.nonzero((smpid == 0) | (smpid == 1))
fix_boundary = list(fix_boundary)
free_boundary, = np.nonzero(smpid > 1)
free_boundary = list(free_boundary)
# find the normal vector for each simplex
simplex_normals = simplex_outward_normals(vert,smp)
# find the normal vectors for each free boundary node
normals = simplex_normals[smpid[free_boundary]]
# add ghost nodes to greatly improve accuracy at the free surface
dx = mindist(nodes)
nodes = np.vstack((nodes,
                   nodes[free_boundary] + dx*normals))

# write out the 2-D equations of motion. This is the most difficult 
# part of setting up the problem and there is no way around it.

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
# body force vector components
f_x = np.zeros(len(interior+free_boundary))
f_y = np.zeros(len(interior+free_boundary)) 
f_z = np.ones(len(interior+free_boundary)) # THIS IS WHERE GRAVITY IS ADDED
f = np.hstack((f_x,f_y,f_z))
# fixed boundary conditions
fix_x = np.zeros(len(fix_boundary))
fix_y = np.zeros(len(fix_boundary))
fix_z = np.zeros(len(fix_boundary))
fix = np.hstack((fix_x,fix_y,fix_z))
# free boundary conditions
free_x = np.zeros(len(free_boundary))
free_y = np.zeros(len(free_boundary))
free_z = np.zeros(len(free_boundary))
free = np.hstack((free_x,free_y,free_z))

## Combine and solve
#####################################################################
# "left hand side" matrix
G = scipy.sparse.vstack((D,dD_fix,dD_free))
G = G.tocsc() # set to csc sparse matrix for efficiency purposes
G.eliminate_zeros()
# "right hand side" vector
d = np.hstack((f,fix,free))
# solve the system of equations
u = scipy.sparse.linalg.spsolve(G,d)
# reshape the solution
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
# compute the second invariant of strain, which is just the sum of 
# each component squared
I2 = np.sqrt(e_xx**2 + e_yy**2 + e_zz**2 + 
             2*e_xy**2 + 2*e_xz**2 + 2*e_yz**2)

## Plot the results
#####################################################################
# toss out ghost nodes
g = len(free_boundary)
nodes = nodes[:-g]
u_x = u_x[:-g]
u_y = u_y[:-g]
u_z = u_z[:-g]
I2 = I2[:-g]

def make_scalar_field(nodes,vals,step=50j,
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


fig = mlab.figure(bgcolor=(1.0,1.0,1.0),fgcolor=(0.0,0.0,0.0),size=(640, 480))
dat = make_scalar_field(nodes,np.log10(I2))
#mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[2:],
#                     color=(1.0,1.0,1.0),opacity=0.2)
#mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp[:2],
#                     color=(1.0,0.0,0.0),opacity=1.0)
mlab.quiver3d(nodes[:,0],nodes[:,1],nodes[:,2],
              u_x,u_y,u_z,mode='arrow',color=(0.2,0.2,0.2))
p = mlab.pipeline.scalar_cut_plane(dat,colormap='hot',vmin=-1,vmax=1)
cbar = mlab.scalarbar(p,title='log10(second strain invariant)')
cbar.title_text_property.bold = False
cbar.title_text_property.italic = False
cbar.label_text_property.bold = False
cbar.label_text_property.italic = False
mlab.show()

#mlab.pipeline.vector_cut_plane(field, scale_factor=.1, colormap='hot')


quit()

