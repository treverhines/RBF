''' 
WORK IN PROGRESS
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import splu, LinearOperator, eigs
from scipy.integrate import solve_ivp

import rbf
from rbf.pde.fd import weight_matrix
from rbf.sputils import expand_rows, expand_cols
from rbf.pde.elastic import (elastic2d_body_force,
                             elastic2d_surface_force,
                             elastic2d_displacement)
from rbf.pde.nodes import poisson_disc_nodes

# define the problem domain
vert = np.array([[0.0, 0.0],
                 [2.0, 0.0],
                 [2.0, 1.0],
                 [1.0, 1.0],
                 [1.0, 2.0],
                 [0.0, 2.0]])
smp = np.array([[0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 0]])
spacing = 0.03
lamb = 1.0
mu = 1.0
rho = 1.0
nu = 1e-5
eval_times = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
plot_eigs = False
stencil_size = 50
order = 4
basis = rbf.basis.phs5

nodes, groups, normals = poisson_disc_nodes(
    spacing,
    (vert, smp),
    boundary_groups={'all':range(len(smp))},
    boundary_groups_with_ghosts=['all'])
print(nodes.shape)    
# create a new node group for my convenience    
groups['interior+boundary:all'] = np.hstack((groups['interior'], 
                                             groups['boundary:all']))
n = nodes.shape[0]    

# We will solve this problem by numerically integrating the state vector `z`.
# The state vector is a flattened concatenation of `u` and `v`. `u` is a (2, n)
# array consisting of:
#
#       * the displacements at the interior at `u[groups['interior']]`
#       * the displacements at the boundary at `u[groups['boundary:all']]`
#       * the boundary conditions as `u[groups['ghosts:all']]`
#
# `v` is a (2, n) array and it is the time derivative of `u`. Here we create
# the initial value for the state vector.
v_init = np.zeros_like(nodes.T)
u_init = np.zeros_like(nodes.T)
# create the initial displacements at the interior and boundary 
r = np.sqrt((nodes[groups['interior+boundary:all'], 0] - 0.5)**2 + 
            (nodes[groups['interior+boundary:all'], 1] - 0.5)**2)
u_init[0, groups['interior+boundary:all']] = 1.0/(1 + (r/0.05)**4)
#u_init[1, groups['interior+boundary:all']] = 1.0/(1 + (r/0.05)**4)
z_init = np.hstack((u_init.flatten(), v_init.flatten()))

# construct a matrix that maps the displacements at `nodes` to `u`
components = elastic2d_displacement(
    nodes[groups['interior+boundary:all']], 
    nodes, 
    n=1)
B_xx = expand_rows(components['xx'], groups['interior+boundary:all'], n)
B_yy = expand_rows(components['yy'], groups['interior+boundary:all'], n)

# this maps displacements to the boundary conditions
components = elastic2d_surface_force(
    nodes[groups['boundary:all']],
    normals[groups['boundary:all']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size,
    order=order,
    phi=basis)
B_xx += expand_rows(components['xx'], groups['ghosts:all'], n)
B_xy  = expand_rows(components['xy'], groups['ghosts:all'], n)
B_yx  = expand_rows(components['yx'], groups['ghosts:all'], n)
B_yy += expand_rows(components['yy'], groups['ghosts:all'], n)

B = sp.vstack((sp.hstack((B_xx, B_xy)),
               sp.hstack((B_yx, B_yy)))).tocsc()
Bsolver = splu(B)

# construct a matrix that maps the displacements at `nodes` to the acceleration
# of `u` due to body forces. 
components = elastic2d_body_force(
    nodes[groups['interior+boundary:all']],
    nodes,
    lamb=lamb,
    mu=mu,
    n=stencil_size,
    order=order,
    phi=basis)
D_xx = expand_rows(components['xx'], groups['interior+boundary:all'], n)
D_xy = expand_rows(components['xy'], groups['interior+boundary:all'], n)
D_yx = expand_rows(components['yx'], groups['interior+boundary:all'], n)
D_yy = expand_rows(components['yy'], groups['interior+boundary:all'], n)

# The boundary conditions are fixed, so there is no acceleration at
# `groups['ghosts:all']`
D = sp.vstack((sp.hstack((D_xx, D_xy)),
               sp.hstack((D_yx, D_yy)))).tocsc()

# construct a matrix that maps `v` to the acceleration of `u` due to
# hyperviscosity. The boundary conditions do not change due to hyperviscosity
# nor do they influence the effect of hyperviscosity on other nodes.
H = weight_matrix(
    nodes[groups['interior+boundary:all']],
    nodes[groups['interior+boundary:all']],
    stencil_size, 
    diffs=[(4, 0), (0, 4)], 
    coeffs=[-1.0, -1.0],
    phi='phs5', 
    order=4)
H = expand_rows(H, groups['interior+boundary:all'], n)     
H = expand_cols(H, groups['interior+boundary:all'], n)     
H = sp.block_diag((H, H)).tocsc()

def state_derivative(t, z):
    u, v = z.reshape((2, -1))
    return np.hstack([v, rho*D.dot(Bsolver.solve(u)) + nu*H.dot(v)])


if plot_eigs:
    L = LinearOperator((4*n, 4*n), matvec=lambda x:state_derivative(0.0, x))
    print('computing eigenvectors')
    vals = eigs(L, 4*n - 2, return_eigenvectors=False)

    print('min real: %s' % np.min(vals.real))
    print('max real: %s' % np.max(vals.real))
    print('min imag: %s' % np.min(vals.imag))
    print('max imag: %s' % np.max(vals.imag))

    plt.plot(vals.real, vals.imag, 'ko')
    plt.grid(ls=':')
    plt.show()

soln = solve_ivp(
    fun=state_derivative, 
    t_span=[eval_times[0], eval_times[-1]],
    y0=z_init,
    method='RK45',
    t_eval=eval_times)

for ti, zi in zip(soln.t, soln.y.T):
    u, v = zi.reshape((2, 2, -1))
    plt.quiver(nodes[:, 0], nodes[:, 1], u[0, :], u[1, :], scale=2.0)
    plt.title('time: %.2f' % ti)
    plt.show()
