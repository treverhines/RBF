''' 
This script demonstrates using the RBF-FD method to calculate 
two-dimensional topographic stresses with roller boundary conditions. 
'''
import numpy as np
from scipy.sparse.linalg import spsolve, gmres
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
import scipy.sparse as sp
from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
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
N_approx = 2000
# size of RBF-FD stencils
n = 20
# Lame parameters
lamb = 1.0
mu = 1.0
#####################################################################
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
groups = {'roller':[0,1,299],
          'free':range(2,299)}
nodes, indices, normals = min_energy_nodes(
                            N_approx, vert, smp, 
                            boundary_groups=groups,
                            boundary_groups_with_ghosts=['roller','free'],
                            rho=node_density)
N = nodes.shape[0]


free = indices['free']
roller = indices['roller']
interior_and_boundary = np.hstack((indices['interior'],indices['roller'],indices['free']))
interior_and_ghost = np.hstack((indices['interior'],indices['roller_ghosts'],indices['free_ghosts']))

# allocate the left-hand-side matrix components
G_xx = sp.csr_matrix((N,N))
G_xy = sp.csr_matrix((N,N))
G_yx = sp.csr_matrix((N,N))
G_yy = sp.csr_matrix((N,N))

# build the "left hand side" matrices for body force constraints
out = elastic2d_body_force(nodes[interior_and_boundary],nodes,
                           lamb=lamb,mu=mu,n=n)
G_xx = add_rows(G_xx, out['xx'], interior_and_ghost)
G_xy = add_rows(G_xy, out['xy'], interior_and_ghost)
G_yx = add_rows(G_yx, out['yx'], interior_and_ghost)
G_yy = add_rows(G_yy, out['yy'], interior_and_ghost)

# build the "left hand side" matrices for free surface constraints
out = elastic2d_surface_force(nodes[free],normals['free'],
                              nodes,lamb=lamb,mu=mu,n=n)
G_xx = add_rows(G_xx, out['xx'], free)
G_xy = add_rows(G_xy, out['xy'], free)
G_yx = add_rows(G_yx, out['yx'], free)
G_yy = add_rows(G_yy, out['yy'], free)

# build the "left hand side" matrices for roller constraints
# constrain displacements in the surface normal direction
out = elastic2d_displacement(nodes[roller],nodes,
                             lamb=lamb,mu=mu,n=1)
normals_x = sp.diags(normals['roller'][:,0])
normals_y = sp.diags(normals['roller'][:,1])
G_xx = add_rows(G_xx, normals_x.dot(out['xx']), roller)
G_xy = add_rows(G_xy, normals_y.dot(out['yy']), roller)

out = elastic2d_surface_force(nodes[roller],normals['roller'],
                              nodes,lamb=lamb,mu=mu,n=n)
parallels = find_orthogonals(normals['roller'])
parallels_x = sp.diags(parallels[:,0])
parallels_y = sp.diags(parallels[:,1])
G_yx = add_rows(G_yx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']), roller)
G_yy = add_rows(G_yy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']), roller)

G_x = sp.hstack((G_xx, G_xy))
G_y = sp.hstack((G_yx, G_yy))
G = sp.vstack((G_x, G_y))
G = G.tocsr()

# form the right-hand-side vector
d_x = np.zeros((N,))
d_y = np.zeros((N,))

d_x[interior_and_ghost] = 0.0
d_x[free] = 0.0
d_x[roller] = 0.0

d_y[interior_and_ghost] = 1.0
d_y[free] = 0.0
d_y[roller] = 0.0

d = np.hstack((d_x,d_y))

u = spsolve(G,d,permc_spec='MMD_ATA')
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
nodes = nodes[interior_and_boundary]
u_x,u_y = u_x[interior_and_boundary],u_y[interior_and_boundary]
e_xx,e_yy,e_xy = e_xx[interior_and_boundary],e_yy[interior_and_boundary],e_xy[interior_and_boundary]
s_xx,s_yy,s_xy = s_xx[interior_and_boundary],s_yy[interior_and_boundary],s_xy[interior_and_boundary]

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


