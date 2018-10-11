''' 
This script demonstrates using the RBF-FD method to calculate static
deformation of a three-dimensional elastic material subject to a
uniform body force such as gravity. The domain has roller boundary
condition on the bottom and sides. The top of domain has a free
surface boundary condition. The top of the domain is intended to
simulate topography.

The linear system is solved with the iterative solver GMRES and
preconditioned with an incomplete LU (ILU) decomposition. 
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
from rbf.domain import topography
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement)
import logging
logging.basicConfig(level=logging.DEBUG)                         


def find_orthogonals(n):
  ''' 
  Returns two arrays of normalized vector that are orthogonal to `n`.
  This is used to determine the directions along which zero traction 
  constraints will be imposed.
    
  Parameters
  ----------
  n : (N,2) float array
    
  Returns
  -------
  out1 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in `n`.

  out2 : (N,2) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in `n` and in `out1`
    
  '''
  out1 = np.empty_like(n,dtype=float)
  out1[:,0] = -np.sum(n,axis=1) + n[:,0]
  out1[:,1:] = n[:,[0]] 
  out1 /= np.linalg.norm(out1,axis=1)[:,None] # normalize length
  out2 = np.cross(n,out1) # find vectors that are orthogonal to *n* and *out1*
  out2 /= np.linalg.norm(out2,axis=1)[:,None] # normalize length
  return out1,out2


## TUNABLE PARAMETERS / FUNCTIONS
#####################################################################
def topo_func(x):
  ''' 
  This is a user-defined function that takes an (N,2) array of surface
  positions and returns an (N,) array of elevations.  
  '''
  r = np.sqrt(x[:,0]**2 + x[:,1]**2)
  return 0.2/(1.0 + (r/0.4)**4)

def density_func(x):
  ''' 
  This function describes the desired node density at `x`. It takes an
  (N,3) array of positions and returns an (N,) array of normalized
  node densities. This function is normalized such that all values are
  between 0.0 and 1.0.
  '''
  r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
  out = 0.1 + 0.9 / (1.0 + (r/0.5)**2)
  return out
  
# number of nodes (excluding ghost nodes)
N = 20000 

# size of RBF-FD stencils.   
n = 50

# Lame parameters
lamb = 1.0
mu = 1.0

# z component of body force 
body_force = 1.0

# this controls the sparsity of the ILU decomposition used for the
# preconditioner. It should be between 0 and 1. smaller values make
# the decomposition denser but better approximates the LU
# decomposition. If the value is too large then you may get a "Factor
# is exactly singular" error.
ilu_drop_tol = 0.005

#####################################################################

## GENERATE THE DOMAIN AND NODES
# generate the domain according to `topo_func` 
vert,smp = topography(topo_func,[-2.0,2.0],[-2.0,2.0],1.0,n=30)

# generate the nodes
boundary_groups = {'free': range(10, smp.shape[0]),
                   'roller': range(0, 10)}
nodes, idx, normals = min_energy_nodes(
                            N,vert,smp,
                            boundary_groups=boundary_groups,
                            boundary_groups_with_ghosts=['free','roller'],
                            rho=density_func,itr=50)
# update `N` to now include the ghost nodes
N = nodes.shape[0]

idx['interior+boundary'] = np.hstack((idx['interior'], 
                                          idx['roller'],
                                          idx['free']))
idx['interior+ghost'] = np.hstack((idx['interior'],
                                   idx['roller_ghosts'],
                                   idx['free_ghosts']))

## BUILD THE LEFT-HAND-SIDE MATRIX
# allocate the left-hand-side matrix components
G_xx = sp.csc_matrix((N,N))
G_xy = sp.csc_matrix((N,N))
G_xz = sp.csc_matrix((N,N))
G_yx = sp.csc_matrix((N,N))
G_yy = sp.csc_matrix((N,N))
G_yz = sp.csc_matrix((N,N))
G_zx = sp.csc_matrix((N,N))
G_zy = sp.csc_matrix((N,N))
G_zz = sp.csc_matrix((N,N))

# add the body force constraints
out = elastic3d_body_force(nodes[idx['interior+boundary']],nodes,
                           lamb=lamb,mu=mu,n=n)
G_xx = add_rows(G_xx, out['xx'], idx['interior+ghost'])
G_xy = add_rows(G_xy, out['xy'], idx['interior+ghost'])
G_xz = add_rows(G_xz, out['xz'], idx['interior+ghost'])
G_yx = add_rows(G_yx, out['yx'], idx['interior+ghost'])
G_yy = add_rows(G_yy, out['yy'], idx['interior+ghost'])
G_yz = add_rows(G_yz, out['yz'], idx['interior+ghost'])
G_zx = add_rows(G_zx, out['zx'], idx['interior+ghost'])
G_zy = add_rows(G_zy, out['zy'], idx['interior+ghost'])
G_zz = add_rows(G_zz, out['zz'], idx['interior+ghost'])

# add free surface constraints
out = elastic3d_surface_force(nodes[idx['free']],normals[idx['free']],
                              nodes,lamb=lamb,mu=mu,n=n)
G_xx = add_rows(G_xx, out['xx'], idx['free'])
G_xy = add_rows(G_xy, out['xy'], idx['free'])
G_xz = add_rows(G_xz, out['xz'], idx['free'])
G_yx = add_rows(G_yx, out['yx'], idx['free'])
G_yy = add_rows(G_yy, out['yy'], idx['free'])
G_yz = add_rows(G_yz, out['yz'], idx['free'])
G_zx = add_rows(G_zx, out['zx'], idx['free'])
G_zy = add_rows(G_zy, out['zy'], idx['free'])
G_zz = add_rows(G_zz, out['zz'], idx['free'])

# add the roller contraints: fixed perpendicular to the surface and
# free parallel to the surface

# fixed normal to the surface
out = elastic3d_displacement(nodes[idx['roller']],nodes,
                             lamb=lamb,mu=mu,n=1)
normals_x = sp.diags(normals[idx['roller']][:,0])
normals_y = sp.diags(normals[idx['roller']][:,1])
normals_z = sp.diags(normals[idx['roller']][:,2])
G_xx = add_rows(G_xx, normals_x.dot(out['xx']), idx['roller'])
G_xy = add_rows(G_xy, normals_y.dot(out['yy']), idx['roller'])
G_xz = add_rows(G_xz, normals_z.dot(out['zz']), idx['roller'])

# free parallel to the surface
out = elastic3d_surface_force(nodes[idx['roller']],normals[idx['roller']],
                              nodes,lamb=lamb,mu=mu,n=n)
parallels_1,parallels_2 = find_orthogonals(normals[idx['roller']])

parallels_x = sp.diags(parallels_1[:,0])
parallels_y = sp.diags(parallels_1[:,1])
parallels_z = sp.diags(parallels_1[:,2])
G_yx = add_rows(G_yx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']) + parallels_z.dot(out['zx']), idx['roller'])
G_yy = add_rows(G_yy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']) + parallels_z.dot(out['zy']), idx['roller'])
G_yz = add_rows(G_yz, parallels_x.dot(out['xz']) + parallels_y.dot(out['yz']) + parallels_z.dot(out['zz']), idx['roller'])

parallels_x = sp.diags(parallels_2[:,0])
parallels_y = sp.diags(parallels_2[:,1])
parallels_z = sp.diags(parallels_2[:,2])
G_zx = add_rows(G_zx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']) + parallels_z.dot(out['zx']), idx['roller'])
G_zy = add_rows(G_zy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']) + parallels_z.dot(out['zy']), idx['roller'])
G_zz = add_rows(G_zz, parallels_x.dot(out['xz']) + parallels_y.dot(out['yz']) + parallels_z.dot(out['zz']), idx['roller'])

# stack the components together. take care to delete matrices when
# we do not need them anymore
del (out, normals_x, normals_y, normals_z, 
     parallels_1, parallels_2, 
     parallels_x, parallels_y, parallels_z) 
 
G_x = sp.hstack((G_xx, G_xy, G_xz))
del G_xx, G_xy, G_xz
G_y = sp.hstack((G_yx, G_yy, G_yz))
del G_yx, G_yy, G_yz
G_z = sp.hstack((G_zx, G_zy, G_zz))
del G_zx, G_zy, G_zz
G = sp.vstack((G_x, G_y, G_z))
del G_x, G_y, G_z
G = G.tocsc()
G.eliminate_zeros()

# create the right-hand-side vector
d_x = np.zeros((N,))
d_y = np.zeros((N,))
d_z = np.zeros((N,))

d_x[idx['interior+ghost']] = 0.0
d_x[idx['free']] = 0.0
d_x[idx['roller']] = 0.0

d_y[idx['interior+ghost']] = 0.0
d_y[idx['free']] = 0.0
d_y[idx['roller']] = 0.0

d_z[idx['interior+ghost']] = body_force
d_z[idx['free']] = 0.0
d_z[idx['roller']] = 0.0

d = np.hstack((d_x, d_y, d_z))

## SOLVE THE SYSTEM
# normalize everything by the norm of the columns of `G`
norm = spla.norm(G,axis=1)
D = sp.diags(1.0/norm) 

d = D.dot(d)
G = D.dot(G).tocsc()

# create the preconditioner with an incomplete LU decomposition of `G`
print('computing ILU decomposition...')
ilu = spla.spilu(G,drop_rule='basic',drop_tol=ilu_drop_tol)
print('done')
M = spla.LinearOperator(G.shape,ilu.solve)

# solve the system using GMRES and define the callback function to
# print info for each iteration
print('solving with GMRES')
def callback(res,_itr=[0]):
  l2 = np.linalg.norm(res)
  print('gmres error on iteration %s: %s' % (_itr[0],l2))
  _itr[0] += 1

u,info = spla.gmres(G,d,M=M,callback=callback)
print('finished gmres with info %s' % info)

## POST-PROCESS THE SOLUTION
del G, d, M, ilu

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

## PLOT THE RESULTS
# figure 1
fig,ax = plt.subplots()
# plot the surface nodes
ax.plot(nodes[idx['free'],0],nodes[idx['free'],1],'C0.',zorder=0)
# plot a contour map of the topography
CS = ax.tricontour(nodes[idx['free'],0],
                   nodes[idx['free'],1],
                   nodes[idx['free'],2],4,colors='k',zorder=1)
plt.clabel(CS, inline=1, fontsize=8)                   
ax.set_aspect('equal')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('surface nodes and topography')
fig.tight_layout()

# figure 2
# plot the a cross section of shear stress
fig,ax = plt.subplots(figsize=(9,4))
x,z = np.meshgrid(np.linspace(-1.0,1.0,200),
                  np.linspace(-1.0,0.0,100))
x = x.flatten()
z = z.flatten()
y = np.zeros_like(x)
points = np.array([x,y,z]).T                   
s_xz_interp = LinearNDInterpolator(nodes,s_xz)(points)
p = ax.tricontourf(x,z,s_xz_interp)
fig.colorbar(p,ax=ax)
ax.set_aspect('equal')
ax.set_xlim(-1.0,1.0)
ax.set_ylim(-1.0,0.0)
ax.set_xlabel('x')
ax.set_ylabel('z (depth)')
ax.set_title('sigma_xz cross section at y=0')
ax.grid(ls=':',color='k')
fig.tight_layout()
plt.show()
