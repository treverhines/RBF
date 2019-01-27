''' 
This script demonstrates using the RBF-FD method to calculate
three-dimensional stress due to tectonic forces and topography. The
tectonic stresses are applied as traction forces to the sides of the
domain. The bottom of the domain has roller boundary conditions,
except there are a few "pinnned" nodes which are required to make the
problem well posed.
'''
import numpy as np
import logging

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.tri

from rbf.basis import phs3
from rbf.geometry import contains
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
    # normalize by the L2 norm of the rows of `G`.
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
  positions and returns an (N,) array of elevations in meters.
  '''
  r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
  return 5000.0/(1.0 + (r/5000.0)**4)


# horizontal tectonic stress components. The vertical component is
# assumed to be zero. The units are kg / m s**2 
tectonic_xx = 100.0e6
tectonic_yy = 0.0
tectonic_xy = 0.0

# Gravitational body force per unit volume. This should be negative
# and have units of kg / m**2 s**2 (density times acceleration)
gravity = -2800.0 * 9.81

# lame parameters. These both have units of kg / m s**2
lamb = 3.2e10 # 32 GPa
mu = 3.2e10 # 32 GPa

# define the domain size in meters
xbounds = np.array([-20.0e3, 20.0e3]) # +/- 20 km
ybounds = np.array([-20.0e3, 20.0e3]) # +/- 20 km
depth = 10.0e3 # 10 km

# The pinned nodes are nodes which have zero displacement. they are
# used to constrain zero translation and zero rotation. We technically
# only need two pinned nodes, but lets use four to keep some symmetry.
# The location of these pinned nodes will affect the solution. They
# should be close to the center of the domain but not so close that
# they effectively become a single point, which will make the problem
# ill-conditioned. 
pinned_nodes = np.array([[ 2500.0,    0.0, -depth],
                         [-2500.0,    0.0, -depth],
                         [    0.0, 2500.0, -depth],
                         [    0.0,-2500.0, -depth]])

# the `nominal_node_count` is the total number of nodes before adding
# ghost nodes
nominal_node_count = 20000
stencil_size = 50
poly_order = 2
basis = phs3

# create the domain boundary as a simplicial complex
vert, smp, boundary_groups = topography(
    topo_func,
    xbounds,
    ybounds,
    depth,
    n=30)
# create the nodes
nodes, groups, normals = min_energy_nodes(
    nominal_node_count, 
    vert, 
    smp, 
    pinned_nodes=pinned_nodes,
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
D_x = weight_matrix(
    nodes, 
    nodes, 
    (1, 0, 0), 
    n=stencil_size, 
    basis=basis, 
    order=poly_order)

D_y = weight_matrix(
    nodes, 
    nodes, 
    (0, 1, 0), 
    n=stencil_size, 
    basis=basis, 
    order=poly_order)

D_z = weight_matrix(
    nodes, 
    nodes, 
    (0, 0, 1), 
    n=stencil_size, 
    basis=basis, 
    order=poly_order)

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

# convert distances from m to km, and stresses from Pa to MPa
nodes /= 1.0e3
vert /= 1.0e3
xbounds /= 1.0e3
ybounds /= 1.0e3
depth /= 1.0e3
s_xx /= 1.0e6
s_yy /= 1.0e6
s_zz /= 1.0e6
s_xy /= 1.0e6
s_xz /= 1.0e6
s_yz /= 1.0e6

# figure 1
fig, ax = plt.subplots()
# plot the surface nodes
ax.plot(
    nodes[groups['boundary:surface'], 0], 
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
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title('surface nodes and topography [km]')
fig.tight_layout()
                                      
# figure 2
def plot_cross_section(stress_component, ax):
    # make a stress cross section along the y=0 plane. This does not
    # extend all the way to the bottom of the domain
    x, z = np.meshgrid(
        np.linspace(*xbounds, 400),
        np.linspace(-0.5*depth, 1.25*np.max(vert[:,2]), 200))
    x = x.flatten()
    z = z.flatten()
    y = np.zeros_like(x)
    points = np.array([x, y, z]).T
    stress = {'xy': s_xy, 'yx': s_xy,
              'zx': s_xz, 'xz': s_xz,
              'yz': s_yz, 'zy': s_yz,
              'zz': s_zz, 'yy': s_yy, 'xx': s_xx}[stress_component]
    stress_interp = LinearNDInterpolator(
        nodes, 
        stress)(points)
    # replace all points that are outside of the domain with the mean
    # stress value. This is done to prevent unexpected color limits in
    # tricontourf
    stress_interp[~contains(points, vert, smp)] = stress.mean()
    # plot stress using tripcolor and mask out points that are outside
    # of the domain. This will show the topography in the cross
    # section
    triang = matplotlib.tri.Triangulation(x, z)
    triang.set_mask(
        ~contains(points[triang.triangles[:,0]], vert, smp) | 
        ~contains(points[triang.triangles[:,1]], vert, smp) | 
        ~contains(points[triang.triangles[:,2]], vert, smp))
    p = ax.tricontourf(
        triang, 
        stress_interp) 
    fig = ax.figure
    cbar = fig.colorbar(p, ax=ax)
    cbar.set_label('stress [MPa]')
    ax.set_aspect('equal')
    ax.set_xlim(*xbounds)
    ax.set_ylim(-0.5*depth, 1.25*np.max(vert[:,2]))
    ax.set_xlabel('x [km]')
    ax.set_ylabel('z (depth) [km]')
    ax.set_title('%s stress cross section at y=0' % stress_component)
    ax.grid(ls=':', color='k')

fig, axs = plt.subplots(3, 2, figsize=(12, 6))
plot_cross_section('xx', axs[0, 0])
plot_cross_section('yy', axs[1, 0])
plot_cross_section('zz', axs[2, 0])
plot_cross_section('xy', axs[0, 1])
plot_cross_section('yz', axs[1, 1])
plot_cross_section('xz', axs[2, 1])

fig.tight_layout()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
# plot a subset of the displacement vectors
idx = np.random.choice(
    np.hstack((groups['interior'], 
               groups['boundary:surface'], 
               groups['boundary:subsurface'])), 
    500, replace=False) 
ax = fig.gca(projection='3d')
ax.quiver(
    nodes[idx, 0], nodes[idx, 1], nodes[idx, 2], 
    u_x[idx], u_y[idx], u_z[idx], 
    length=0.1)
ax.set_title('displacement')    
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
plt.show()

#
