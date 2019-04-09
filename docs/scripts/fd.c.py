''' 
This script demonstrates using the RBF-FD method to calculate static
deformation of a three-dimensional elastic material subject to a
uniform body force such as gravity. The elastic material has a fixed
boundary condition on one side and the remaining sides have a free
surface boundary condition.  This script also demonstrates using ghost
nodes which, for all intents and purposes, are necessary when dealing
with Neumann boundary conditions.
'''
import logging

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rbf.sputils import add_rows
from rbf.linalg import GMRESSolver
from rbf.pde.nodes import min_energy_nodes
from rbf.pde.fd import weight_matrix
from rbf.pde.elastic import (elastic3d_body_force,
                             elastic3d_surface_force,
                             elastic3d_displacement) 

logging.basicConfig(level=logging.DEBUG)

## User defined parameters
#####################################################################
# define the vertices of the problem domain. Note that the first two
# simplices will be fixed, and the others will be free
vert = np.array([[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 1.0, 1.0],
                 [2.0, 0.0, 0.0],
                 [2.0, 0.0, 1.0],
                 [2.0, 1.0, 0.0],
                 [2.0, 1.0, 1.0]])
smp = np.array([[0, 1, 3],
                [0, 2, 3],
                [0, 1, 4],
                [1, 5, 4],
                [0, 2, 6],
                [0, 4, 6],
                [1, 7, 5],
                [1, 3, 7],
                [4, 5, 7],
                [4, 6, 7],
                [2, 3, 7],
                [2, 6, 7]])
# number of nodes 
N = 500
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
    boundary_groups_with_ghosts=['free'])
N = nodes.shape[0]

# The "left hand side" matrices are built with the convenience
# functions from *rbf.elastic*. Read the documentation for these
# functions to better understand this step.
G_xx = sp.coo_matrix((N, N))
G_xy = sp.coo_matrix((N, N))
G_xz = sp.coo_matrix((N, N))

G_yx = sp.coo_matrix((N, N))
G_yy = sp.coo_matrix((N, N))
G_yz = sp.coo_matrix((N, N))

G_zx = sp.coo_matrix((N, N))
G_zy = sp.coo_matrix((N, N))
G_zz = sp.coo_matrix((N, N))

out = elastic3d_body_force(nodes[idx['interior']], nodes, 
                           lamb=lamb, mu=mu)
G_xx = add_rows(G_xx, out['xx'], idx['interior'])
G_xy = add_rows(G_xy, out['xy'], idx['interior'])
G_xz = add_rows(G_xz, out['xz'], idx['interior'])
G_yx = add_rows(G_yx, out['yx'], idx['interior'])
G_yy = add_rows(G_yy, out['yy'], idx['interior'])
G_yz = add_rows(G_yz, out['yz'], idx['interior'])
G_zx = add_rows(G_zx, out['zx'], idx['interior'])
G_zy = add_rows(G_zy, out['zy'], idx['interior'])
G_zz = add_rows(G_zz, out['zz'], idx['interior'])

out = elastic3d_body_force(nodes[idx['boundary:free']], nodes, 
                           lamb=lamb, mu=mu)
G_xx = add_rows(G_xx, out['xx'], idx['ghosts:free'])
G_xy = add_rows(G_xy, out['xy'], idx['ghosts:free'])
G_xz = add_rows(G_xz, out['xz'], idx['ghosts:free'])
G_yx = add_rows(G_yx, out['yx'], idx['ghosts:free'])
G_yy = add_rows(G_yy, out['yy'], idx['ghosts:free'])
G_yz = add_rows(G_yz, out['yz'], idx['ghosts:free'])
G_zx = add_rows(G_zx, out['zx'], idx['ghosts:free'])
G_zy = add_rows(G_zy, out['zy'], idx['ghosts:free'])
G_zz = add_rows(G_zz, out['zz'], idx['ghosts:free'])

out = elastic3d_surface_force(nodes[idx['boundary:free']], 
                              normals[idx['boundary:free']], 
                              nodes, lamb=lamb, mu=mu)
G_xx = add_rows(G_xx, out['xx'], idx['boundary:free'])
G_xy = add_rows(G_xy, out['xy'], idx['boundary:free'])
G_xz = add_rows(G_xz, out['xz'], idx['boundary:free'])
G_yx = add_rows(G_yx, out['yx'], idx['boundary:free'])
G_yy = add_rows(G_yy, out['yy'], idx['boundary:free'])
G_yz = add_rows(G_yz, out['yz'], idx['boundary:free'])
G_zx = add_rows(G_zx, out['zx'], idx['boundary:free'])
G_zy = add_rows(G_zy, out['zy'], idx['boundary:free'])
G_zz = add_rows(G_zz, out['zz'], idx['boundary:free'])

out = elastic3d_displacement(nodes[idx['boundary:fix']], nodes)
G_xx = add_rows(G_xx, out['xx'], idx['boundary:fix'])
G_yy = add_rows(G_yy, out['yy'], idx['boundary:fix'])
G_zz = add_rows(G_zz, out['zz'], idx['boundary:fix'])

G_x = sp.hstack((G_xx, G_xy, G_xz))
G_y = sp.hstack((G_yx, G_yy, G_yz))
G_z = sp.hstack((G_zx, G_zy, G_zz))
G = sp.vstack((G_x, G_y, G_z))

# build the right-hand-side vector
d_x = np.zeros((N,))
d_y = np.zeros((N,))
d_z = np.ones((N,))

d_x[idx['interior']] = 0.0
d_x[idx['ghosts:free']] = 0.0
d_x[idx['boundary:free']] = 0.0
d_x[idx['boundary:fix']] = 0.0

d_y[idx['interior']] = 0.0
d_y[idx['ghosts:free']] = 0.0
d_y[idx['boundary:free']] = 0.0
d_y[idx['boundary:fix']] = 0.0

d_z[idx['interior']] = body_force
d_z[idx['ghosts:free']] = body_force
d_z[idx['boundary:free']] = 0.0
d_z[idx['boundary:fix']] = 0.0

d = np.hstack((d_x, d_y, d_z))

# solve it
u = GMRESSolver(G).solve(d)
u = np.reshape(u,(3,-1))
u_x,u_y,u_z = u

## Plot the results
#####################################################################
idx_int_and_bnd = np.hstack((idx['interior'],
                             idx['boundary:free'],
                             idx['boundary:fix']))

nodes = nodes[idx_int_and_bnd]
u_x,u_y,u_z = (u_x[idx_int_and_bnd],
               u_y[idx_int_and_bnd],
               u_z[idx_int_and_bnd])

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
