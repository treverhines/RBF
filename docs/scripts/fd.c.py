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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement) 
import logging
logging.basicConfig(level=logging.DEBUG)

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
N = 500
# size of RBF-FD stencils
n = 30
# lame parameters
lamb = 1.0
mu = 1.0
# z component of body force
body_force = 1.0

## Build and solve for displacements and strain
#####################################################################
# generate nodes. Note that this may take a while
boundary_groups = {'fix':[0,1],
                   'free':range(2,12)}
nodes, idx, normals = min_energy_nodes(
                        N,vert,smp,
                        boundary_groups=boundary_groups,
                        boundary_groups_with_ghosts=['free'],
                        include_vertices=True)
N = nodes.shape[0]
idx['interior+free'] = np.hstack((idx['interior'],
                                  idx['free']))
idx['interior+ghosts'] = np.hstack((idx['interior'],
                                    idx['free_ghosts']))
idx['interior+boundary'] = np.hstack((idx['interior'],
                                      idx['free'],
                                      idx['fix']))

# The "left hand side" matrices are built with the convenience
# functions from *rbf.fdbuild*. Read the documentation for these
# functions to better understand this step.
G_xx = sp.csr_matrix((N, N))
G_xy = sp.csr_matrix((N, N))
G_xz = sp.csr_matrix((N, N))

G_yx = sp.csr_matrix((N, N))
G_yy = sp.csr_matrix((N, N))
G_yz = sp.csr_matrix((N, N))

G_zx = sp.csr_matrix((N, N))
G_zy = sp.csr_matrix((N, N))
G_zz = sp.csr_matrix((N, N))

out = elastic3d_body_force(nodes[idx['interior+free']], nodes, 
                           lamb=lamb, mu=mu, n=n)
G_xx = add_rows(G_xx, out['xx'], idx['interior+ghosts'])
G_xy = add_rows(G_xy, out['xy'], idx['interior+ghosts'])
G_xz = add_rows(G_xz, out['xz'], idx['interior+ghosts'])
G_yx = add_rows(G_yx, out['yx'], idx['interior+ghosts'])
G_yy = add_rows(G_yy, out['yy'], idx['interior+ghosts'])
G_yz = add_rows(G_yz, out['yz'], idx['interior+ghosts'])
G_zx = add_rows(G_zx, out['zx'], idx['interior+ghosts'])
G_zy = add_rows(G_zy, out['zy'], idx['interior+ghosts'])
G_zz = add_rows(G_zz, out['zz'], idx['interior+ghosts'])

out = elastic3d_surface_force(nodes[idx['free']], normals[idx['free']], nodes, 
                              lamb=lamb, mu=mu, n=n)
G_xx = add_rows(G_xx, out['xx'], idx['free'])
G_xy = add_rows(G_xy, out['xy'], idx['free'])
G_xz = add_rows(G_xz, out['xz'], idx['free'])
G_yx = add_rows(G_yx, out['yx'], idx['free'])
G_yy = add_rows(G_yy, out['yy'], idx['free'])
G_yz = add_rows(G_yz, out['yz'], idx['free'])
G_zx = add_rows(G_zx, out['zx'], idx['free'])
G_zy = add_rows(G_zy, out['zy'], idx['free'])
G_zz = add_rows(G_zz, out['zz'], idx['free'])

out = elastic3d_displacement(nodes[idx['fix']], nodes, 
                             lamb=lamb, mu=mu, n=1)
G_xx = add_rows(G_xx, out['xx'], idx['fix'])
G_yy = add_rows(G_yy, out['yy'], idx['fix'])
G_zz = add_rows(G_zz, out['zz'], idx['fix'])

G_x = sp.hstack((G_xx, G_xy, G_xz))
G_y = sp.hstack((G_yx, G_yy, G_yz))
G_z = sp.hstack((G_zx, G_zy, G_zz))
G = sp.vstack((G_x, G_y, G_z))
G = G.tocsr()

# build the right-hand-side vector
d_x = np.zeros((N,))
d_y = np.zeros((N,))
d_z = np.ones((N,))

d_x[idx['interior+ghosts']] = 0.0
d_x[idx['free']] = 0.0
d_x[idx['fix']] = 0.0

d_y[idx['interior+ghosts']] = 0.0
d_y[idx['free']] = 0.0
d_y[idx['fix']] = 0.0

d_z[idx['interior+ghosts']] = body_force
d_z[idx['free']] = 0.0
d_z[idx['fix']] = 0.0

d = np.hstack((d_x, d_y, d_z))

# solve it
u = spla.spsolve(G,d,permc_spec='MMD_ATA')
u = np.reshape(u,(3,-1))
u_x,u_y,u_z = u

## Plot the results
#####################################################################
nodes = nodes[idx['interior+boundary']]
u_x,u_y,u_z = (u_x[idx['interior+boundary']],
               u_y[idx['interior+boundary']],
               u_z[idx['interior+boundary']])

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_aspect('equal')

ax.plot_trisurf(vert[:,0],vert[:,1],vert[:,2],triangles=smp,
                color=(0.1,0.1,0.1),  
                shade=False,alpha=0.2)
ax.quiver(nodes[:,0], nodes[:,1], nodes[:,2], u_x, u_y, u_z,
          length=0.01, color='k')
          
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
