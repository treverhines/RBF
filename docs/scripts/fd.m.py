''' 
This script demonstrates using the RBF-FD method to calculate
two-dimensional stress
'''
import numpy as np
import logging

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

import rbf
from rbf.nodes import min_energy_nodes
from rbf.fd import weight_matrix, add_rows
from rbf.domain import topography
from rbf.fdbuild import (elastic3d_body_force,
                         elastic3d_surface_force,
                         elastic3d_displacement)

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


def find_orthogonals(n):
  ''' 
  Returns two arrays of normalized vector that are orthogonal to `n`.
  This is used to determine the directions along which zero traction 
  constraints will be imposed.
    
  Parameters
  ----------
  n : (N, 3) float array
    
  Returns
  -------
  out1 : (N, 3) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in `n`.

  out2 : (N, 3) float array
    Array of normalized vectors that are orthogonal to the 
    corresponding vectors in `n` and in `out1`
    
  '''
  out1 = np.empty_like(n, dtype=float)
  out1[:, 0] = -n[:, 1] - n[:, 2]
  out1[:, 1] = n[:, 0]
  out1[:, 2] = n[:, 0]
  out1 /= np.linalg.norm(out1, axis=1)[:, None]
  out2 = np.cross(n, out1) 
  out2 /= np.linalg.norm(out2, axis=1)[:, None] 
  return out1, out2


def topo_func(x):
  ''' 
  This is a user-defined function that takes an (N,2) array of surface
  positions and returns an (N,) array of elevations.  
  '''
  r = np.sqrt(x[:,0]**2 + x[:,1]**2)
  return 0*0.2/(1.0 + (r/0.4)**4)

# horizontal tectonic stress components. The vertical component is
# assumed to be zero.
tectonic_xx = -1.0
tectonic_yy = 1.0
tectonic_xy = 0.0

# gravity should be negative
gravity = 0.0

# define the domain and user parameters
xbounds = [-2.0, 2.0]
ybounds = [-2.0, 2.0]
depth = 1.0

# The pinned nodes are nodes which have zero displacement. they are
# used to constrain zero translation and zero rotation. We technically
# only need two pinned nodes, but lets use five to keep some symmetry
pinned_nodes = [[  0.0,  0.0, -1.0],
                [ 1.00,  0.0, -1.0],
                [-1.00,  0.0, -1.0],
                [  0.0, 1.00, -1.0],
                [  0.0,-1.00, -1.0]]

# the `nominal_node_count` is the total number of nodes before adding
# ghost nodes
nominal_node_count = 50000
stencil_size = 50
poly_order = 2
basis = rbf.basis.phs3
# lame parameters
lamb = 1.0
mu = 1.0

# create the nodes
vert, smp, boundary_groups = topography(
    topo_func,
    xbounds,
    ybounds,
    depth,
    n=20)

nodes, groups, normals = min_energy_nodes(
    nominal_node_count, 
    vert, 
    smp, 
    itr=100,
    pinned_nodes=pinned_nodes,
    include_vertices=False,
    boundary_groups=boundary_groups,
    boundary_groups_with_ghosts=['surface', 'sides', 'bottom'])
node_count = nodes.shape[0]

# allocate the left-hand-side matrix components
G_xx = sp.csc_matrix((node_count, node_count))
G_xy = sp.csc_matrix((node_count, node_count))
G_xz = sp.csc_matrix((node_count, node_count))

G_yx = sp.csc_matrix((node_count, node_count))
G_yy = sp.csc_matrix((node_count, node_count))
G_yz = sp.csc_matrix((node_count, node_count))

G_zx = sp.csc_matrix((node_count, node_count))
G_zy = sp.csc_matrix((node_count, node_count))
G_zz = sp.csc_matrix((node_count, node_count))

# build the "left hand side" matrices for body force constraints on
# the interior nodes
out = elastic3d_body_force(
    nodes[groups['interior']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    basis=basis,
    order=poly_order)

G_xx = add_rows(G_xx, out['xx'], groups['interior'])
G_xy = add_rows(G_xy, out['xy'], groups['interior'])
G_xz = add_rows(G_xz, out['xz'], groups['interior'])

G_yx = add_rows(G_yx, out['yx'], groups['interior'])
G_yy = add_rows(G_yy, out['yy'], groups['interior'])
G_yz = add_rows(G_yz, out['yz'], groups['interior'])

G_zx = add_rows(G_zx, out['zx'], groups['interior'])
G_zy = add_rows(G_zy, out['zy'], groups['interior'])
G_zz = add_rows(G_zz, out['zz'], groups['interior'])

# enforce body force constraints at the boundaries using the rows
# corresponding to the ghost nodes
for k in ['surface', 'sides', 'bottom']:
    out = elastic3d_body_force(
        nodes[groups['boundary:' + k]], 
        nodes, 
        lamb=lamb, 
        mu=mu, 
        n=stencil_size,
        basis=basis,
        order=poly_order)

    G_xx = add_rows(G_xx, out['xx'], groups['ghosts:' + k])
    G_xy = add_rows(G_xy, out['xy'], groups['ghosts:' + k])
    G_xz = add_rows(G_xz, out['xz'], groups['ghosts:' + k])

    G_yx = add_rows(G_yx, out['yx'], groups['ghosts:' + k])
    G_yy = add_rows(G_yy, out['yy'], groups['ghosts:' + k])
    G_yz = add_rows(G_yz, out['yz'], groups['ghosts:' + k])

    G_zx = add_rows(G_zx, out['zx'], groups['ghosts:' + k])
    G_zy = add_rows(G_zy, out['zy'], groups['ghosts:' + k])
    G_zz = add_rows(G_zz, out['zz'], groups['ghosts:' + k])


# build the "left hand side" matrices for traction force constraints
# on the surface and the sides. These are used to enforce a free
# surface and tectonic stresses
for k in ['surface', 'sides']:
    out = elastic3d_surface_force(
        nodes[groups['boundary:' + k]], 
        normals[groups['boundary:' + k]], 
        nodes, 
        lamb=lamb, 
        mu=mu, 
        n=stencil_size,
        basis=basis,
        order=poly_order)

    G_xx = add_rows(G_xx, out['xx'], groups['boundary:' + k])
    G_xy = add_rows(G_xy, out['xy'], groups['boundary:' + k])
    G_xz = add_rows(G_xz, out['xz'], groups['boundary:' + k])

    G_yx = add_rows(G_yx, out['yx'], groups['boundary:' + k])
    G_yy = add_rows(G_yy, out['yy'], groups['boundary:' + k])
    G_yz = add_rows(G_yz, out['yz'], groups['boundary:' + k])

    G_zx = add_rows(G_zx, out['zx'], groups['boundary:' + k])
    G_zy = add_rows(G_zy, out['zy'], groups['boundary:' + k])
    G_zz = add_rows(G_zz, out['zz'], groups['boundary:' + k])


# build the "left hand side" matrices for roller constraints on the
# bottom
normals_x = sp.diags(normals[groups['boundary:bottom']][:, 0])
normals_y = sp.diags(normals[groups['boundary:bottom']][:, 1])
normals_z = sp.diags(normals[groups['boundary:bottom']][:, 2])

parallels1, parallels2 = find_orthogonals(normals[groups['boundary:bottom']])

parallels1_x = sp.diags(parallels1[:, 0])
parallels1_y = sp.diags(parallels1[:, 1])
parallels1_z = sp.diags(parallels1[:, 2])

parallels2_x = sp.diags(parallels2[:, 0])
parallels2_y = sp.diags(parallels2[:, 1])
parallels2_z = sp.diags(parallels2[:, 2])

# have zero displacement normal to the boundary
out = elastic3d_displacement(
    nodes[groups['boundary:bottom']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=1,
    basis=basis,
    order=0)

# [G_xx, G_xy, G_xz] * [u_x, u_y, u_z] should be the displacement
# along the normal direction
G_xx = add_rows(G_xx, normals_x.dot(out['xx']), groups['boundary:bottom'])
G_xy = add_rows(G_xy, normals_y.dot(out['yy']), groups['boundary:bottom'])
G_xz = add_rows(G_xz, normals_z.dot(out['zz']), groups['boundary:bottom'])

# have zero traction parallel to the boundary
out = elastic3d_surface_force(
    nodes[groups['boundary:bottom']], 
    normals[groups['boundary:bottom']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=stencil_size,
    basis=basis,
    order=poly_order)

# [G_yx, G_yy, G_yz] * [u_x, u_y, u_z] should be one of the traction
# components that is parallel to the surface
G_yx = add_rows(
    G_yx, 
    parallels1_x.dot(out['xx']) + 
    parallels1_y.dot(out['yx']) +
    parallels1_z.dot(out['zx']), 
    groups['boundary:bottom'])                
G_yy = add_rows(
    G_yy, 
    parallels1_x.dot(out['xy']) + 
    parallels1_y.dot(out['yy']) + 
    parallels1_z.dot(out['zy']),
    groups['boundary:bottom'])                                
G_yz = add_rows(
    G_yz, 
    parallels1_x.dot(out['xz']) + 
    parallels1_y.dot(out['yz']) + 
    parallels1_z.dot(out['zz']),
    groups['boundary:bottom'])

# [G_zx, G_zy, G_zz] * [u_x, u_y, u_z] should be the other traction
# components that is parallel to the surface
G_zx = add_rows(
    G_zx, 
    parallels2_x.dot(out['xx']) + 
    parallels2_y.dot(out['yx']) +
    parallels2_z.dot(out['zx']), 
    groups['boundary:bottom'])                
G_zy = add_rows(
    G_zy, 
    parallels2_x.dot(out['xy']) + 
    parallels2_y.dot(out['yy']) + 
    parallels2_z.dot(out['zy']),
    groups['boundary:bottom'])                                
G_zz = add_rows(
    G_zz, 
    parallels2_x.dot(out['xz']) + 
    parallels2_y.dot(out['yz']) + 
    parallels2_z.dot(out['zz']),
    groups['boundary:bottom'])

# add the constraint that displacement at the pinned node is zero.
out = elastic3d_displacement(
    nodes[groups['pinned']], 
    nodes, 
    lamb=lamb, 
    mu=mu, 
    n=1,
    basis=basis,
    order=0)
G_xx = add_rows(G_xx, out['xx'], groups['pinned'])
G_yy = add_rows(G_yy, out['yy'], groups['pinned'])
G_zz = add_rows(G_zz, out['zz'], groups['pinned'])

# stack it all together, removing unneeded matrices as soon as
# possible
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

# form the right-hand-side vector
d_x = np.zeros((node_count,))
d_y = np.zeros((node_count,))
d_z = np.zeros((node_count,))

d_x[groups['interior']] = 0.0
d_x[groups['ghosts:surface']] = 0.0
d_x[groups['ghosts:sides']] = 0.0
d_x[groups['ghosts:bottom']] = 0.0
d_x[groups['boundary:surface']] = 0.0
# The x component of the boundary normal vectors dotted with the
# tectonic stress tensor
d_x[groups['boundary:sides']] = (
    normals[groups['boundary:sides'], 0]*tectonic_xx +
    normals[groups['boundary:sides'], 1]*tectonic_xy)
d_x[groups['boundary:bottom']] = 0.0
d_x[groups['pinned']] = 0.0

d_y[groups['interior']] = 0.0
d_y[groups['ghosts:surface']] = 0.0
d_y[groups['ghosts:sides']] = 0.0
d_y[groups['ghosts:bottom']] = 0.0
d_y[groups['boundary:surface']] = 0.0
# The y component of the boundary normal vectors dotted with the
# tectonic stress tensor
d_y[groups['boundary:sides']] = (
    normals[groups['boundary:sides'], 0]*tectonic_xy +
    normals[groups['boundary:sides'], 1]*tectonic_yy)
d_y[groups['boundary:bottom']] = 0.0
d_y[groups['pinned']] = 0.0

d_z[groups['interior']] = -gravity
d_z[groups['ghosts:surface']] = -gravity
d_z[groups['ghosts:sides']] = -gravity
d_z[groups['ghosts:bottom']] = -gravity
d_z[groups['boundary:surface']] = 0.0
d_z[groups['boundary:sides']] = 0.0
d_z[groups['boundary:bottom']] = 0.0
d_z[groups['pinned']] = 0.0

d = np.hstack((d_x, d_y, d_z))

# solve the system using an ILU decomposition and GMRES
u = gmres_ilu_solve(G, d)
u = np.reshape(u,(3, -1))
u_x, u_y, u_z = u

# Calculate strain and stress from displacements
D_x = weight_matrix(nodes, nodes, (1, 0, 0), n=stencil_size, order=poly_order)
D_y = weight_matrix(nodes, nodes, (0, 1, 0), n=stencil_size, order=poly_order)
D_z = weight_matrix(nodes, nodes, (0, 0, 1), n=stencil_size, order=poly_order)

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
fig, ax = plt.subplots()
# plot the surface nodes
ax.plot(nodes[groups['boundary:surface'], 0], 
        nodes[groups['boundary:surface'], 1], 
        'C0.', zorder=0)
# plot a contour map of the topography
CS = ax.tricontour(
    nodes[groups['boundary:surface'], 0],
    nodes[groups['boundary:surface'], 1],
    nodes[groups['boundary:surface'], 2], 
    4, colors='k', zorder=1)
plt.clabel(CS, inline=1, fontsize=8)
ax.set_aspect('equal')
ax.set_xlim(*xbounds)
ax.set_ylim(*ybounds)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('surface nodes and topography')
fig.tight_layout()
                                      
# figure 2
def plot_cross_section(stress_component, ax):
    # make a stress cross section along the y=0 plane
    x, z = np.meshgrid(np.linspace(*xbounds, 200),
                       np.linspace(-depth, 0.0, 100))
    x = x.flatten()
    z = z.flatten()
    y = np.zeros_like(x)
    points = np.array([x, y, z]).T
    stress = {'xy': s_xy, 'yx': s_xy,
              'zx': s_xz, 'xz': s_xz,
              'yz': s_yz, 'zy': s_yz,
              'zz': s_zz, 'yy': s_yy, 'xx': s_xx}[stress_component]
    stress_interp = LinearNDInterpolator(nodes, stress)(points)

    fig = ax.figure
    p = ax.tricontourf(x, z, stress_interp)
    fig.colorbar(p, ax=ax)
    ax.set_aspect('equal')
    ax.set_xlim(*xbounds)
    ax.set_ylim(-depth, 0.0)
    ax.set_xlabel('x')
    ax.set_ylabel('z (depth)')
    ax.set_title('%s stress cross section at y=0' % stress_component)
    ax.grid(ls=':', color='k')

fig, axs = plt.subplots(3, 2, figsize=(14, 6))
plot_cross_section('xx', axs[0, 0])
plot_cross_section('yy', axs[1, 0])
plot_cross_section('zz', axs[2, 0])
plot_cross_section('xy', axs[0, 1])
plot_cross_section('yz', axs[1, 1])
plot_cross_section('xz', axs[2, 1])

fig.tight_layout()
plt.show()

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.quiver(nodes[:,0],nodes[:,1],nodes[:,2], u_x, u_y, u_z, length=1.0)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()
#
