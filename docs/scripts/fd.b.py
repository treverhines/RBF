''' 
This script demonstrates using the RBF-FD method to calculate static
deformation of a two-dimensional elastic material subject to a uniform body
force such as gravity. The elastic material has a fixed boundary condition on
one side and the remaining sides have a free surface boundary condition.  This
script also demonstrates using ghost nodes which, for all intents and purposes,
are necessary when dealing with Neumann boundary conditions.
'''
import logging

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rbf.sputils import expand_rows
from rbf.linalg import GMRESSolver
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes

logging.basicConfig(level=logging.DEBUG)

## User defined parameters
#####################################################################
# define the vertices of the problem domain. Note that the first simplex will
# be fixed, and the others will be free
vert = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
# The node spacing
dx = 0.05
# size of RBF-FD stencils
n = 30
# lame parameters
lamb = 1.0
mu = 1.0
# z component of body for
body_force = 1.0

## Build and solve for displacements and strain
#####################################################################
# generate nodes. The nodes are assigned groups based on which simplex they lay
# on
boundary_groups = {'fixed':[0], 'free':[1, 2, 3]}
nodes, groups, normals = poisson_disc_nodes(
    dx, 
    (vert, smp),
    boundary_groups=boundary_groups,
    boundary_groups_with_ghosts=['free'])
N = nodes.shape[0]
# `nodes` : (N, 2) float array
# `groups` : dictionary containing index sets. It has the keys
#            "interior", "boundary:free", "boundary:fixed",
#            "ghosts:free".
# `normals : (N, 2) float array

## Enforce the PDE on interior nodes AND the free surface nodes 
# x component of force resulting from displacement in the x direction.
coeffs_xx = [lamb+2*mu, mu]
diffs_xx = [(2, 0), (0, 2)]
# x component of force resulting from displacement in the y direction.
coeffs_xy = [lamb, mu]
diffs_xy = [(1, 1), (1, 1)]
# y component of force resulting from displacement in the x direction.
coeffs_yx = [mu, lamb]
diffs_yx = [(1, 1), (1, 1)]
# y component of force resulting from displacement in the y direction.
coeffs_yy = [lamb+2*mu, mu]
diffs_yy =  [(0, 2), (2, 0)]
# make the differentiation matrices that enforce the PDE on the interior nodes.
D_xx = weight_matrix(nodes[groups['interior']], nodes, n, diffs_xx, coeffs=coeffs_xx)
D_xy = weight_matrix(nodes[groups['interior']], nodes, n, diffs_xy, coeffs=coeffs_xy)
D_yx = weight_matrix(nodes[groups['interior']], nodes, n, diffs_yx, coeffs=coeffs_yx)
D_yy = weight_matrix(nodes[groups['interior']], nodes, n, diffs_yy, coeffs=coeffs_yy)
G_xx = expand_rows(D_xx, groups['interior'], N)
G_xy = expand_rows(D_xy, groups['interior'], N)
G_yx = expand_rows(D_yx, groups['interior'], N)
G_yy = expand_rows(D_yy, groups['interior'], N)

# use the ghost nodes to enforce the PDE on the boundary
D_xx = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_xx, coeffs=coeffs_xx)
D_xy = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_xy, coeffs=coeffs_xy)
D_yx = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_yx, coeffs=coeffs_yx)
D_yy = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_yy, coeffs=coeffs_yy)
G_xx += expand_rows(D_xx, groups['ghosts:free'], N)
G_xy += expand_rows(D_xy, groups['ghosts:free'], N)
G_yx += expand_rows(D_yx, groups['ghosts:free'], N)
G_yy += expand_rows(D_yy, groups['ghosts:free'], N)

## Enforce fixed boundary conditions
# Enforce that x and y are as specified with the fixed boundary condition.
# These matrices turn out to be identity matrices, but I include this
# computation for consistency with the rest of the code. feel free to comment
# out the next couple lines and replace it with an appropriately sized sparse
# identity matrix.
coeffs_xx = [1.0]
diffs_xx = [(0, 0)]
coeffs_yy = [1.0]
diffs_yy = [(0, 0)]

dD_fix_xx = weight_matrix(nodes[groups['boundary:fixed']], nodes, n, diffs_xx, coeffs=coeffs_xx)
dD_fix_yy = weight_matrix(nodes[groups['boundary:fixed']], nodes, n, diffs_yy, coeffs=coeffs_yy)
G_xx += expand_rows(dD_fix_xx, groups['boundary:fixed'], N)
G_yy += expand_rows(dD_fix_yy, groups['boundary:fixed'], N)

## Enforce free surface boundary conditions
# x component of traction force resulting from x displacement 
coeffs_xx = [normals[groups['boundary:free']][:, 0]*(lamb+2*mu), 
             normals[groups['boundary:free']][:, 1]*mu]
diffs_xx = [(1, 0), (0, 1)]
# x component of traction force resulting from y displacement
coeffs_xy = [normals[groups['boundary:free']][:, 0]*lamb, 
             normals[groups['boundary:free']][:, 1]*mu]
diffs_xy = [(0, 1), (1, 0)]
# y component of traction force resulting from x displacement
coeffs_yx = [normals[groups['boundary:free']][:, 0]*mu, 
             normals[groups['boundary:free']][:, 1]*lamb]
diffs_yx = [(0, 1), (1, 0)]
# y component of force resulting from displacement in the y direction
coeffs_yy = [normals[groups['boundary:free']][:, 1]*(lamb+2*mu), 
             normals[groups['boundary:free']][:, 0]*mu]
diffs_yy =  [(0, 1), (1, 0)]
# make the differentiation matrices that enforce the free surface boundary 
# conditions.
dD_free_xx = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_xx, coeffs=coeffs_xx)
dD_free_xy = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_xy, coeffs=coeffs_xy)
dD_free_yx = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_yx, coeffs=coeffs_yx)
dD_free_yy = weight_matrix(nodes[groups['boundary:free']], nodes, n, diffs_yy, coeffs=coeffs_yy)

G_xx += expand_rows(dD_free_xx, groups['boundary:free'], N)
G_xy += expand_rows(dD_free_xy, groups['boundary:free'], N)
G_yx += expand_rows(dD_free_yx, groups['boundary:free'], N)
G_yy += expand_rows(dD_free_yy, groups['boundary:free'], N)

# stack the components together to form the left-hand-side matrix
G_x = sp.hstack((G_xx, G_xy))
G_y = sp.hstack((G_yx, G_yy))
G = sp.vstack((G_x, G_y))

# form the right-hand-side vector
d_x = np.zeros((N,))
d_y = np.zeros((N,))

d_x[groups['interior']] = 0.0
d_x[groups['ghosts:free']] = 0.0
d_x[groups['boundary:free']] = 0.0
d_x[groups['boundary:fixed']] = 0.0

d_y[groups['interior']] = body_force
d_y[groups['ghosts:free']] = body_force
d_y[groups['boundary:free']] = 0.0
d_y[groups['boundary:fixed']] = 0.0

d = np.hstack((d_x, d_y))

# solve the system!
u = GMRESSolver(G).solve(d)

# reshape the solution
u = np.reshape(u, (2, -1))
u_x, u_y = u

## Calculate strain on a fine grid from displacements
x, y = np.meshgrid(np.linspace(0.0, 2.0, 100), np.linspace(0.0, 1.0, 50))
points = np.array([x.flatten(), y.flatten()]).T

D_x = weight_matrix(points, nodes, n, (1, 0))
D_y = weight_matrix(points, nodes, n, (0, 1))
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
# calculate second strain invariant
I2 = np.sqrt(e_xx**2 + e_yy**2 + 2*e_xy**2)

## Plot the results
#####################################################################
idx_no_ghosts = np.hstack((groups['interior'],
                           groups['boundary:free'],
                           groups['boundary:fixed']))
                          
nodes = nodes[idx_no_ghosts]
u_x = u_x[idx_no_ghosts]
u_y = u_y[idx_no_ghosts]

fig, ax = plt.subplots(figsize=(7, 3.5))
# plot the fixed boundary
ax.plot(vert[smp[0], 0], vert[smp[0], 1], 'r-', lw=2, label='fixed', zorder=1)
# plot the free boundary
ax.plot(vert[smp[1], 0], vert[smp[1], 1], 'r--', lw=2, label='free', zorder=1)
for s in smp[2:]:
  ax.plot(vert[s, 0], vert[s, 1], 'r--', lw=2, zorder=1)

# plot the second strain invariant
p = ax.tripcolor(points[:, 0], points[:, 1], I2,
                 norm=LogNorm(vmin=0.1, vmax=3.2),
                 cmap='viridis', zorder=0)
# plot the displacement vectors
ax.quiver(nodes[:, 0], nodes[:, 1], u_x, u_y, zorder=2)
ax.set_xlim((-0.1, 2.1))
ax.set_ylim((-0.25, 1.1))
ax.set_aspect('equal')
ax.legend(loc=3, frameon=False, fontsize=12, ncol=2)
cbar = fig.colorbar(p)
cbar.set_label('second strain invariant')
fig.tight_layout()
plt.savefig('../figures/fd.b.png')
plt.show() 
