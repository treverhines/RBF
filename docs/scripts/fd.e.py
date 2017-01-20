''' 
This script demonstrates using the RBF-FD method to calculate 
two-dimensional topographic stresses with roller boundary conditions. 
'''
import numpy as np
from scipy.sparse import vstack,hstack,diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from rbf.nodes import menodes
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force,
                         elastic2d_displacement)

####################### USER PARAMETERS #############################
scale = 10.0
def node_density(x):
  c = np.array([0.0,1.0])
  r = (x[:,0] - c[0])**2 + (x[:,1] - c[1])**2
  out = 0.75/(1.0 + (r/2.0)**2)
  out += 0.25/(1.0 + (r/(2*scale))**2)
  return out

vert_bot = np.array([[scale,-scale],
                     [-scale,-scale]])
vert_top_x = np.linspace(-scale,scale,298)
vert_top_y = 1.0/(1 + vert_top_x**2) # define topography
vert_top = np.array([vert_top_x,vert_top_y]).T
vert = np.vstack((vert_bot,vert_top))
smp = np.array([np.arange(300),np.roll(np.arange(300),-1)]).T
# number of nodes 
N = 5000
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
  
def find_orthogonals(n):
  ''' 
  Returns an array of normalized vector that are orthogonal to *n*. 
  This is used to determine the directions along which zero traction 
  constraints will be imposed.
    
  Parameters
  ----------
  n : (N,2) float array
    
  Returns
  -------
  out : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vector in *n*.
    
  '''
  out = np.empty_like(n,dtype=float)
  out[:,0] = -np.sum(n,axis=1) + n[:,0]
  out[:,1:] = n[:,[0]]
  out /= np.linalg.norm(out,axis=1)[:,None]
  return out

# generate nodes. Note that this may take a while
nodes,smpid = menodes(N,vert,smp,rho=node_density)
# roller nodes are on the bottom and sides of the domain
roller_idx = np.nonzero((smpid == 0) |
                        (smpid == 1) |
                        (smpid == 299))[0].tolist()
# free nodes are on the top of the domain
free_idx = np.nonzero(~((smpid == -1) |
                        (smpid == 0)  |
                        (smpid == 1)  |
                        (smpid == 299)))[0].tolist()
# interior nodes
int_idx = np.nonzero(smpid == -1)[0].tolist()
# find normal vectors for each simplex
simplex_normals = simplex_outward_normals(vert,smp)
# find normal vectors for each boundary node
roller_normals = simplex_normals[smpid[roller_idx]]
free_normals = simplex_normals[smpid[free_idx]]
# surface parallel vectors
roller_parallels = find_orthogonals(roller_normals)
# add ghost nodes next to free surface and roller nodes
dx = min_distance(nodes)
nodes = np.vstack((nodes,nodes[roller_idx] + dx*roller_normals))
nodes = np.vstack((nodes,nodes[free_idx] + dx*free_normals))
# build the "left hand side" matrices for body force constraints
A_body = elastic2d_body_force(nodes[int_idx+free_idx+roller_idx],nodes,lamb=lamb,mu=mu,n=n)
A_body_x,A_body_y = (hstack(i) for i in A_body)
# build the "right hand side" vectors for body force constraints
b_body_x = np.zeros_like(int_idx+free_idx+roller_idx)
b_body_y = np.ones_like(int_idx+free_idx+roller_idx) # THIS IS WHERE GRAVITY IS IMPOSED
# build the "left hand side" matrices for free surface constraints
A_surf = elastic2d_surface_force(nodes[free_idx],free_normals,nodes,lamb=lamb,mu=mu,n=n)
A_surf_x,A_surf_y = (hstack(i) for i in A_surf)
# build the "right hand side" vectors for free surface constraints
b_surf_x = np.zeros_like(free_idx)
b_surf_y = np.zeros_like(free_idx)
# build the "left hand side" matrices for roller constraints
# constrain displacements in the surface normal direction
A_roller_disp = elastic2d_displacement(nodes[roller_idx],nodes,lamb=lamb,mu=mu,n=1)
A_roller_disp_x,A_roller_disp_y = (hstack(i) for i in A_roller_disp)
normals_x = diags(roller_normals[:,0])
normals_y = diags(roller_normals[:,1])
A_roller_n = normals_x.dot(A_roller_disp_x) + normals_y.dot(A_roller_disp_y)
# constrain surface traction in the surface parallel direction
A_roller_surf = elastic2d_surface_force(nodes[roller_idx],roller_normals,nodes,lamb=lamb,mu=mu,n=n)
A_roller_surf_x,A_roller_surf_y = (hstack(i) for i in A_roller_surf)
parallels_x = diags(roller_parallels[:,0])
parallels_y = diags(roller_parallels[:,1])
A_roller_p = parallels_x.dot(A_roller_surf_x) + parallels_y.dot(A_roller_surf_y)
# build the "right hand side" vectors for roller constraints
b_roller_n = np.zeros_like(roller_idx)
b_roller_p = np.zeros_like(roller_idx)
# stack it all together and solve
A = vstack((A_body_x,A_body_y,
            A_surf_x,A_surf_y,
            A_roller_n,A_roller_p)).tocsr()
b = np.hstack((b_body_x,b_body_y,
               b_surf_x,b_surf_y,
               b_roller_n,b_roller_p))
u = spsolve(A,b,permc_spec='MMD_ATA')
u = np.reshape(u,(2,-1))
u_x,u_y = u
# Calculate strain and stress from displacements
D_x = weight_matrix(nodes,nodes,(1,0),n=n)
D_y = weight_matrix(nodes,nodes,(0,1),n=n)
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
s_xx = (2*mu + lamb)*e_xx + lamb*e_yy
s_yy = lamb*e_xx + (2*mu + lamb)*e_yy
s_xy = 2*mu*e_xy

# plot the results
#####################################################################
def grid(x, y, z):
    points = np.array([x,y]).T
    xg,yg = np.mgrid[min(x):max(x):1000j,min(y):max(y):1000j]
    zg = griddata(points,z,(xg,yg),method='linear')
    return xg,yg,zg

# toss out ghosts
nodes = nodes[:N]
u_x,u_y = u_x[:N],u_y[:N]
e_xx,e_yy,e_xy = e_xx[:N],e_yy[:N],e_xy[:N]
s_xx,s_yy,s_xy = s_xx[:N],s_yy[:N],s_xy[:N]

fig,axs = plt.subplots(2,2,figsize=(10,7))
poly = Polygon(vert,facecolor='none',edgecolor='k',zorder=3)
axs[0][0].add_artist(poly)
poly = Polygon(vert,facecolor='none',edgecolor='k',zorder=3)
axs[0][1].add_artist(poly)
poly = Polygon(vert,facecolor='none',edgecolor='k',zorder=3)
axs[1][0].add_artist(poly)
poly = Polygon(vert,facecolor='none',edgecolor='k',zorder=3)
axs[1][1].add_artist(poly)
# flip the bottom vertices to make a mask polygon
vert[vert[:,-1] < 0.0] *= -1 
poly = Polygon(vert,facecolor='w',edgecolor='k',zorder=3)
axs[0][0].add_artist(poly)
poly = Polygon(vert,facecolor='w',edgecolor='k',zorder=3)
axs[0][1].add_artist(poly)
poly = Polygon(vert,facecolor='w',edgecolor='k',zorder=3)
axs[1][0].add_artist(poly)
poly = Polygon(vert,facecolor='w',edgecolor='k',zorder=3)
axs[1][1].add_artist(poly)

axs[0][0].quiver(nodes[:,0],nodes[:,1],u_x,u_y,scale=1000.0,width=0.005)
axs[0][0].set_xlim((0,3))
axs[0][0].set_ylim((-2,1))
axs[0][0].set_aspect('equal')
axs[0][0].set_title('displacements',fontsize=10)

xg,yg,s_xyg = grid(nodes[:,0],nodes[:,1],s_xx)
p = axs[0][1].contourf(xg,yg,s_xyg,np.arange(-1.0,0.04,0.04),cmap='viridis',zorder=1)
axs[0][1].set_xlim((0,3))
axs[0][1].set_ylim((-2,1))
axs[0][1].set_aspect('equal')
cbar = fig.colorbar(p,ax=axs[0][1])
cbar.set_label('sigma_xx',fontsize=10)

xg,yg,s_xyg = grid(nodes[:,0],nodes[:,1],s_yy)
p = axs[1][0].contourf(xg,yg,s_xyg,np.arange(-2.6,0.2,0.2),cmap='viridis',zorder=1)
axs[1][0].set_xlim((0,3))
axs[1][0].set_ylim((-2,1))
axs[1][0].set_aspect('equal')
cbar = fig.colorbar(p,ax=axs[1][0])
cbar.set_label('sigma_yy',fontsize=10)

xg,yg,s_xyg = grid(nodes[:,0],nodes[:,1],s_xy)
p = axs[1][1].contourf(xg,yg,s_xyg,np.arange(0.0,0.24,0.02),cmap='viridis',zorder=1)
axs[1][1].set_xlim((0,3))
axs[1][1].set_ylim((-2,1))
axs[1][1].set_aspect('equal')
cbar = fig.colorbar(p,ax=axs[1][1])
cbar.set_label('sigma_xy',fontsize=10)

plt.show()


