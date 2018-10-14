''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a two-dimensional elastic material subject to a uniform 
body force such as gravity. The elastic material has a fixed boundary 
condition on one side and the remaining sides have a free surface 
boundary condition.  This script also demonstrates using ghost nodes 
which, for all intents and purposes, are necessary when dealing with 
Neumann boundary conditions.
'''
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
import scipy.sparse.linalg as spla
import logging
logging.basicConfig(level=logging.DEBUG)

## User defined parameters
#####################################################################
# define the vertices of the problem domain. Note that the first 
# simplex will be fixed, and the others will be free
vert = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
# The number of nodes excluding ghost nodes
N = 1000
# size of RBF-FD stencils
n = 20
# lame parameters
lamb = 1.0
mu = 1.0
# z component of body force
body_force = 1.0

## Build and solve for displacements and strain
#####################################################################
# generate nodes. The nodes are assigned groups based on which simplex
# they lay on
boundary_groups = {'fixed':[0],
                   'free':[1, 2, 3]}
nodes, groups, normals = min_energy_nodes(
  N, vert, smp,
  boundary_groups=boundary_groups,
  boundary_groups_with_ghosts=['free'])
# `nodes` : (N, 2) float array
# `groups` : dictionary containing index sets. It has the keys
#            "interior", "boundary:free", "boundary:fixed",
#            "ghosts:free".
# `normals : (N, 2) float array

# update `N` to include ghost nodes
N = nodes.shape[0]

## Create the sparse submatrices for the system matrix 
G_xx = sp.csr_matrix((N, N))
G_xy = sp.csr_matrix((N, N))
G_yx = sp.csr_matrix((N, N))
G_yy = sp.csr_matrix((N, N))

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
# make the differentiation matrices that enforce the PDE on the 
# interior nodes.
D_xx = weight_matrix(nodes[groups['interior']], nodes, diffs_xx, coeffs=coeffs_xx, n=n)
D_xy = weight_matrix(nodes[groups['interior']], nodes, diffs_xy, coeffs=coeffs_xy, n=n)
D_yx = weight_matrix(nodes[groups['interior']], nodes, diffs_yx, coeffs=coeffs_yx, n=n)
D_yy = weight_matrix(nodes[groups['interior']], nodes, diffs_yy, coeffs=coeffs_yy, n=n)
G_xx = add_rows(G_xx, D_xx, groups['interior'])
G_xy = add_rows(G_xy, D_xy, groups['interior'])
G_yx = add_rows(G_yx, D_yx, groups['interior'])
G_yy = add_rows(G_yy, D_yy, groups['interior'])

# use the ghost nodes to enforce the PDE on the boundary
D_xx = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_xx, coeffs=coeffs_xx, n=n)
D_xy = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_xy, coeffs=coeffs_xy, n=n)
D_yx = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_yx, coeffs=coeffs_yx, n=n)
D_yy = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_yy, coeffs=coeffs_yy, n=n)
G_xx = add_rows(G_xx, D_xx, groups['ghosts:free'])
G_xy = add_rows(G_xy, D_xy, groups['ghosts:free'])
G_yx = add_rows(G_yx, D_yx, groups['ghosts:free'])
G_yy = add_rows(G_yy, D_yy, groups['ghosts:free'])

## Enforce fixed boundary conditions
# Enforce that x and y are as specified with the fixed boundary 
# condition. These matrices turn out to be identity matrices, but I 
# include this computation for consistency with the rest of the code. 
# feel free to comment out the next couple lines and replace it with 
# an appropriately sized sparse identity matrix.
coeffs_xx = [1.0]
diffs_xx = [(0, 0)]
coeffs_yy = [1.0]
diffs_yy = [(0, 0)]

dD_fix_xx = weight_matrix(nodes[groups['boundary:fixed']], nodes, diffs_xx, coeffs=coeffs_xx, n=n)
dD_fix_yy = weight_matrix(nodes[groups['boundary:fixed']], nodes, diffs_yy, coeffs=coeffs_yy, n=n)
G_xx = add_rows(G_xx, dD_fix_xx, groups['boundary:fixed'])
G_yy = add_rows(G_yy, dD_fix_yy, groups['boundary:fixed'])

## Enforce free surface boundary conditions
# x component of traction force resulting from x displacement 
coeffs_xx = [normals[groups['boundary:free']][:, 0]*(lamb+2*mu), normals[groups['boundary:free']][:, 1]*mu]
diffs_xx = [(1, 0), (0, 1)]
# x component of traction force resulting from y displacement
coeffs_xy = [normals[groups['boundary:free']][:, 0]*lamb, normals[groups['boundary:free']][:, 1]*mu]
diffs_xy = [(0, 1), (1, 0)]
# y component of traction force resulting from x displacement
coeffs_yx = [normals[groups['boundary:free']][:, 0]*mu, normals[groups['boundary:free']][:, 1]*lamb]
diffs_yx = [(0, 1), (1, 0)]
# y component of force resulting from displacement in the y direction
coeffs_yy = [normals[groups['boundary:free']][:, 1]*(lamb+2*mu), normals[groups['boundary:free']][:, 0]*mu]
diffs_yy =  [(0, 1), (1, 0)]
# make the differentiation matrices that enforce the free surface boundary 
# conditions.
dD_free_xx = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_xx, coeffs=coeffs_xx, n=n)
dD_free_xy = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_xy, coeffs=coeffs_xy, n=n)
dD_free_yx = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_yx, coeffs=coeffs_yx, n=n)
dD_free_yy = weight_matrix(nodes[groups['boundary:free']], nodes, diffs_yy, coeffs=coeffs_yy, n=n)

G_xx = add_rows(G_xx, dD_free_xx, groups['boundary:free'])
G_xy = add_rows(G_xy, dD_free_xy, groups['boundary:free'])
G_yx = add_rows(G_yx, dD_free_yx, groups['boundary:free'])
G_yy = add_rows(G_yy, dD_free_yy, groups['boundary:free'])

# stack the components together to form the left-hand-side matrix
G_x = sp.hstack((G_xx, G_xy))
G_y = sp.hstack((G_yx, G_yy))
G = sp.vstack((G_x, G_y))
G = G.tocsr()

# for the right-hand-side vector
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
u = sp.linalg.spsolve(G, d)

# reshape the solution
u = np.reshape(u, (2, -1))
u_x, u_y = u
## Calculate strain from displacements
D_x = weight_matrix(nodes, nodes, (1, 0), n=n)
D_y = weight_matrix(nodes, nodes, (0, 1), n=n)
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
I2 = I2[idx_no_ghosts]

fig, ax = plt.subplots(figsize=(7, 3.5))
# plot the fixed boundary
ax.plot(vert[smp[0], 0], vert[smp[0], 1], 'r-', lw=2, label='fixed', zorder=1)
# plot the free boundary
ax.plot(vert[smp[1], 0], vert[smp[1], 1], 'r--', lw=2, label='free', zorder=1)
for s in smp[2:]:
  ax.plot(vert[s, 0], vert[s, 1], 'r--', lw=2, zorder=1)

# plot the second strain invariant
p = ax.tripcolor(nodes[:, 0], nodes[:, 1], I2,
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
