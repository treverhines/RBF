''' 
This script demonstrates using the RBF-FD method to calculate
two-dimensional stress with tectonic stress boundary conditions
'''
import numpy as np
import logging

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import spsolve, gmres
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
from rbf.fdbuild import (elastic2d_body_force,
                         elastic2d_surface_force,
                         elastic2d_displacement)

logging.basicConfig(level=logging.DEBUG)


def gmres_ilu_solve(G, d, drop_tol=0.005):
    '''
    Solves the sparse linear system using GMRES and an incomplete LU
    factorization as a preconditioner

    Parameters
    ----------
    G : (n, n) sparse matrix

    d : (n,) array

    drop_tol : float (optional)
        this controls the sparsity of the ILU decomposition used for
        the preconditioner. It should be between 0 and 1. smaller
        values make the decomposition denser but better approximates
        the LU decomposition. If the value is too large then you may
        get a "Factor is exactly singular" error.

    Returns
    -------
    (n,) array
    '''
    # normalize everything by the norm of the columns of `G`
    norm = spla.norm(G, axis=1)
    D = sp.diags(1.0/norm)

    d = D.dot(d)
    G = D.dot(G).tocsc()

    # create the preconditioner with an ILU decomposition of `G`
    print('computing ILU decomposition...')
    ilu = spla.spilu(G, drop_rule='basic', drop_tol=drop_tol)
    print('done')
    M = spla.LinearOperator(G.shape, ilu.solve)

    # solve the system using GMRES and define the callback function to
    # print info for each iteration
    def callback(res, _itr=[0]):
        l2 = np.linalg.norm(res)
        print('gmres error on iteration %s: %s' % (_itr[0], l2))
        _itr[0] += 1

    print('solving with GMRES')
    u, info = spla.gmres(G, d, M=M, callback=callback)
    print('finished gmres with info %s' % info)
    return u


# define the domain and user parameters
vert = [[0.0, 0.0],
        [10.0, 0.0],
        [10.0, 2.0],
        [6.0, 2.0],
        [5.0, 2.2],
        [4.0, 2.0],
        [0.0, 2.0]]
smp = [[0, 1],
       [1, 2],
       [2, 3],
       [3, 4],
       [4, 5],
       [5, 6],
       [6, 0]]
pinned_node = [5.0, 0.0]
groups = {'bottom': [0],
          'top': [2, 3, 4, 5],
          'sides': [1, 6]}

nominal_node_count = 1000
stencil_size = 30
poly_order = 2
# lame parameters
lamb = 1.0
mu = 1.0

# create the nodes
nodes, idx, normals = min_energy_nodes(
  nominal_node_count, 
  vert, 
  smp, 
  itr=1000,
  pinned_nodes=[pinned_node],
  include_vertices=False,
  boundary_groups=groups,
  boundary_groups_with_ghosts=['top', 'bottom', 'sides'])

node_count = nodes.shape[0]

for k, v in idx.items():
    plt.plot(nodes[v, 0], nodes[v, 1], 'o', label=k, ms=5)
    plt.quiver(nodes[v, 0], nodes[v, 1], normals[v, 0], normals[v, 1])

plt.legend()
plt.show()    

# allocate the left-hand-side matrix components
G_xx = sp.csc_matrix((node_count, node_count))
G_xy = sp.csc_matrix((node_count, node_count))
G_yx = sp.csc_matrix((node_count, node_count))
G_yy = sp.csc_matrix((node_count, node_count))

# build the "left hand side" matrices for body force constraints
out = elastic2d_body_force(
    nodes[idx['interior']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['interior'])
G_xy = add_rows(G_xy, out['xy'], idx['interior'])
G_yx = add_rows(G_yx, out['yx'], idx['interior'])
G_yy = add_rows(G_yy, out['yy'], idx['interior'])

out = elastic2d_body_force(
    nodes[idx['boundary:top']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['ghosts:top'])
G_xy = add_rows(G_xy, out['xy'], idx['ghosts:top'])
G_yx = add_rows(G_yx, out['yx'], idx['ghosts:top'])
G_yy = add_rows(G_yy, out['yy'], idx['ghosts:top'])

out = elastic2d_body_force(
    nodes[idx['boundary:bottom']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['ghosts:bottom'])
G_xy = add_rows(G_xy, out['xy'], idx['ghosts:bottom'])
G_yx = add_rows(G_yx, out['yx'], idx['ghosts:bottom'])
G_yy = add_rows(G_yy, out['yy'], idx['ghosts:bottom'])

out = elastic2d_body_force(
    nodes[idx['boundary:sides']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['ghosts:sides'])
G_xy = add_rows(G_xy, out['xy'], idx['ghosts:sides'])
G_yx = add_rows(G_yx, out['yx'], idx['ghosts:sides'])
G_yy = add_rows(G_yy, out['yy'], idx['ghosts:sides'])

# build the "left hand side" matrices for traction force constraints
# on the tops and the sides
out = elastic2d_surface_force(
    nodes[idx['boundary:top']], 
    normals[idx['boundary:top']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['boundary:top'])
G_xy = add_rows(G_xy, out['xy'], idx['boundary:top'])
G_yx = add_rows(G_yx, out['yx'], idx['boundary:top'])
G_yy = add_rows(G_yy, out['yy'], idx['boundary:top'])

out = elastic2d_surface_force(
    nodes[idx['boundary:sides']], 
    normals[idx['boundary:sides']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
G_xx = add_rows(G_xx, out['xx'], idx['boundary:sides'])
G_xy = add_rows(G_xy, out['xy'], idx['boundary:sides'])
G_yx = add_rows(G_yx, out['yx'], idx['boundary:sides'])
G_yy = add_rows(G_yy, out['yy'], idx['boundary:sides'])


# build the "left hand side" matrices for roller constraints
normals_x = sp.diags(normals[idx['boundary:bottom']][:, 0])
normals_y = sp.diags(normals[idx['boundary:bottom']][:, 1])

parallels_x = sp.diags(normals[idx['boundary:bottom']][:, 1])
parallels_y = sp.diags(-normals[idx['boundary:bottom']][:, 0])

# have zero displacement normal to the boundary
out = elastic2d_displacement(
    nodes[idx['boundary:bottom']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=1,
    order=0)
# [G_xx, G_xy] * [u_x, u_y] is the displacement along the normal
# direction
G_xx = add_rows(G_xx, normals_x.dot(out['xx']), idx['boundary:bottom'])
G_xy = add_rows(G_xy, normals_y.dot(out['yy']), idx['boundary:bottom'])

# have zero traction parallel to the boundary
out = elastic2d_surface_force(
    nodes[idx['boundary:bottom']], 
    normals[idx['boundary:bottom']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    order=poly_order)
# [G_yx, G_yy] * [u_x, u_y] is the traction component that is parallel
# to the surface
G_yx = add_rows(G_yx, parallels_x.dot(out['xx']) + parallels_y.dot(out['yx']), idx['boundary:bottom'])
G_yy = add_rows(G_yy, parallels_x.dot(out['xy']) + parallels_y.dot(out['yy']), idx['boundary:bottom'])

# add the constraint that displacement at the pinned node is zero
G_xx[idx['pinned'], idx['pinned']] = 1.0
G_yy[idx['pinned'], idx['pinned']] = 1.0


G_x = sp.hstack((G_xx, G_xy))
G_y = sp.hstack((G_yx, G_yy))
G = sp.vstack((G_x, G_y))
G = G.tocsc()
G.eliminate_zeros()

# remove any excess matrices to free up memory for the ILU
# decomposition
del G_x, G_y, G_xx, G_xy, G_yx, G_yy


# form the right-hand-side vector
d_x = np.zeros((node_count,))
d_y = np.zeros((node_count,))

d_x[idx['interior']] = 0.0
d_x[idx['ghosts:top']] = 0.0
d_x[idx['ghosts:bottom']] = 0.0
d_x[idx['ghosts:sides']] = 0.0
d_x[idx['boundary:top']] = 0.0
d_x[idx['boundary:bottom']] = 0.0
d_x[idx['boundary:sides']] = -1.0*normals[idx['boundary:sides'], 0]

d_y[idx['interior']] = 1.0
d_y[idx['ghosts:top']] = 1.0
d_y[idx['ghosts:bottom']] = 1.0
d_y[idx['ghosts:sides']] = 1.0
d_y[idx['boundary:top']] = 0.0
d_y[idx['boundary:bottom']] = 0.0
d_y[idx['boundary:sides']] = 0.0

d = np.hstack((d_x, d_y))

# solve the system using an ILU decomposition and GMRES
u = gmres_ilu_solve(G, d)
#u = spsolve(G, d)
u = np.reshape(u,(2,-1))
u_x, u_y = u

# Calculate strain and stress from displacements
D_x = weight_matrix(nodes, nodes, (1, 0), n=stencil_size, order=poly_order)
D_y = weight_matrix(nodes, nodes, (0, 1), n=stencil_size, order=poly_order)

e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))

s_xx = (2*mu + lamb)*e_xx + lamb*e_yy
s_yy = lamb*e_xx + (2*mu + lamb)*e_yy
s_xy = 2*mu*e_xy

## plot the results
######################################################################

keep = np.hstack((idx['interior'], 
                  idx['boundary:top'],
                  idx['boundary:bottom'],
                  idx['boundary:sides']))
u_x = u_x[keep]
u_y = u_y[keep]
s_xx = s_xx[keep]
s_xy = s_xy[keep]
s_yy = s_yy[keep]
nodes = nodes[keep]

fig, ax = plt.subplots()
p = ax.tripcolor(nodes[:, 0], nodes[:, 1], s_xx)
ax.set_title('s_xx')
ax.set_aspect('equal')
fig.colorbar(p)

fig, ax = plt.subplots()
p = ax.tripcolor(nodes[:, 0], nodes[:, 1], s_xy)
ax.set_title('s_xy')
ax.set_aspect('equal')
fig.colorbar(p)

fig, ax = plt.subplots()
p = ax.tripcolor(nodes[:, 0], nodes[:, 1], s_yy)
ax.set_title('s_yy')
ax.set_aspect('equal')
fig.colorbar(p)

fig, ax = plt.subplots()
plt.quiver(nodes[:, 0], nodes[:, 1], u_x, u_y)
plt.show()

#def grid(x, y, z):
#    points = np.array([x,y]).T
#    xg,yg = np.mgrid[min(x):max(x):1000j,min(y):max(y):1000j]
#    zg = griddata(points,z,(xg,yg),method='linear')
#    return xg,yg,zg
#
## toss out ghosts
#idx_no_ghosts = np.hstack((idx['interior'],
#                           idx['boundary:roller'],
#                           idx['boundary:free']))
#
#nodes = nodes[idx_no_ghosts]
#u_x, u_y = u_x[idx_no_ghosts], u_y[idx_no_ghosts]
#e_xx, e_yy, e_xy = e_xx[idx_no_ghosts], e_yy[idx_no_ghosts], e_xy[idx_no_ghosts]
#s_xx, s_yy, s_xy = s_xx[idx_no_ghosts], s_yy[idx_no_ghosts], s_xy[idx_no_ghosts]
#
#fig, axs = plt.subplots(2, 2, figsize=(10, 7))
#poly = Polygon(vert, facecolor='none', edgecolor='k', zorder=3)
#axs[0][0].add_artist(poly)
#poly = Polygon(vert, facecolor='none', edgecolor='k', zorder=3)
#axs[0][1].add_artist(poly)
#poly = Polygon(vert, facecolor='none', edgecolor='k', zorder=3)
#axs[1][0].add_artist(poly)
#poly = Polygon(vert, facecolor='none', edgecolor='k', zorder=3)
#axs[1][1].add_artist(poly)
## flip the bottom vertices to make a mask polygon
#vert[vert[:, -1] < 0.0] *= -1 
#poly = Polygon(vert, facecolor='w', edgecolor='k', zorder=3)
#axs[0][0].add_artist(poly)
#poly = Polygon(vert, facecolor='w', edgecolor='k', zorder=3)
#axs[0][1].add_artist(poly)
#poly = Polygon(vert, facecolor='w', edgecolor='k', zorder=3)
#axs[1][0].add_artist(poly)
#poly = Polygon(vert, facecolor='w', edgecolor='k', zorder=3)
#axs[1][1].add_artist(poly)
#
#axs[0][0].quiver(nodes[:,0], nodes[:,1], u_x, u_y, scale=1000.0, width=0.005)
#axs[0][0].set_xlim((0, 3))
#axs[0][0].set_ylim((-2, 1))
#axs[0][0].set_aspect('equal')
#axs[0][0].set_title('displacements', fontsize=10)
#
#xg, yg, s_xxg = grid(nodes[:,0], nodes[:,1], s_xx)
#p = axs[0][1].contourf(xg, yg, s_xxg, np.arange(-1.0, 0.04, 0.04), cmap='viridis', zorder=1)
#axs[0][1].set_xlim((0,3))
#axs[0][1].set_ylim((-2,1))
#axs[0][1].set_aspect('equal')
#cbar = fig.colorbar(p,ax=axs[0][1])
#cbar.set_label('sigma_xx',fontsize=10)
#
#xg, yg, s_yyg = grid(nodes[:,0], nodes[:,1], s_yy)
#p = axs[1][0].contourf(xg, yg, s_yyg, np.arange(-2.6, 0.2, 0.2), cmap='viridis', zorder=1)
#axs[1][0].set_xlim((0, 3))
#axs[1][0].set_ylim((-2, 1))
#axs[1][0].set_aspect('equal')
#cbar = fig.colorbar(p, ax=axs[1][0])
#cbar.set_label('sigma_yy', fontsize=10)
#
#xg, yg, s_xyg = grid(nodes[:,0], nodes[:,1], s_xy)
#p = axs[1][1].contourf(xg, yg, s_xyg, np.arange(0.0, 0.24, 0.02), cmap='viridis',zorder=1)
#axs[1][1].set_xlim((0, 3))
#axs[1][1].set_ylim((-2, 1))
#axs[1][1].set_aspect('equal')
#cbar = fig.colorbar(p, ax=axs[1][1])
#cbar.set_label('sigma_xy', fontsize=10)
#
#plt.tight_layout()
#plt.show()
#
#
#n
