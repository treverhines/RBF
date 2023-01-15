'''
This script demonstrates using RBF-FD to solve the heat equation with spatially
variable conductivity, k(x, y). Namely,

    u_t = k_x*u_x + k*u_xx + k_y*u_y + k*u_yy

'''
import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp

from rbf.pde.nodes import poisson_disc_nodes
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
from rbf.sputils import expand_rows, expand_cols

logging.basicConfig(level=logging.DEBUG)

node_spacing = 0.1
radial_basis_function = 'phs3' # 3rd order polyharmonic spline
polynomial_order = 2
stencil_size = 30
time_evaluations = np.linspace(0, 1, 41)

vertices = np.array(
    [[-2.    ,  2.    ], [ 2.    ,  2.    ], [ 2.    , -2.    ],
     [-2.    , -2.    ], [ 1.    ,  0.    ], [ 0.9239,  0.3827],
     [ 0.7071,  0.7071], [ 0.3827,  0.9239], [ 0.    ,  1.    ],
     [-0.3827,  0.9239], [-0.7071,  0.7071], [-0.9239,  0.3827],
     [-1.    ,  0.    ], [-0.9239, -0.3827], [-0.7071, -0.7071],
     [-0.3827, -0.9239], [-0.    , -1.    ], [ 0.3827, -0.9239],
     [ 0.7071, -0.7071], [ 0.9239, -0.3827]]
    )

simplices = np.array(
    [[0 , 1 ], [1 , 2 ], [2 , 3 ], [3 , 0 ], [4 , 5 ], [5 , 6 ], [6 , 7 ],
     [7 , 8 ], [8 , 9 ], [9 , 10], [10, 11], [11, 12], [12, 13], [13, 14],
     [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19,  4]]
    )

boundary_groups = {
    'fixed': [1, 3],
    'free': [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }

def conductivity(x):
    out = 1 + 3*(1 / (1 + np.exp(-5*x[:, 1])))
    return out

def conductivity_xdiff(x):
    # x derivative of conductivity
    out = (conductivity(x + [0.01, 0.0]) - conductivity(x))/0.01
    return out

def conductivity_ydiff(x):
    # y derivative of conductivity
    out = (conductivity(x + [0.0, 0.01]) - conductivity(x))/0.01
    return out

# Generate the nodes. `groups` identifies which group each node belongs to.
# `normals` contains the normal vectors corresponding to each node (nan if the
# node is not on a boundary). We are giving the free boundary nodes ghost
# nodes to improve the accuracy of the free boundary constraint.
nodes, grp, normals = poisson_disc_nodes(
    node_spacing,
    (vertices, simplices),
    boundary_groups=boundary_groups,
    boundary_groups_with_ghosts=['free']
    )

# We will solve the heat equation by numerically integrating the state vector
# `z`. The state vector consists of:
#   - The temperature of the interior nodes at z[grp['interior']]
#   - The temperature of the fixed boundary at z[grp['boundary:fixed']]
#   - The temperature of the free boundary at z[grp['boundary:free']]
#   - The boundary condition at the free boundary at z[grp['ghosts:free']]
#
z_init = np.zeros(nodes.shape[0])
z_init[grp['boundary:fixed']] = nodes[grp['boundary:fixed'], 0]

# Create a matrix that maps the temperature at `nodes` to the state vector.
# Then create a solver for it, so we can map the state vector to the
# temperature at all nodes.

# `idx` are the indices for all nodes except the ghost nodes. The mapping to
# the state vector is just an identity mapping so we only need a stencil size
# of 1.
idx = np.hstack((grp['interior'], grp['boundary:fixed'], grp['boundary:free']))
B_temp = weight_matrix(
    x=nodes[idx],
    p=nodes,
    n=1,
    diffs=[0, 0]
    )
B_free = weight_matrix(
    x=nodes[grp['boundary:free']],
    p=nodes,
    n=stencil_size,
    diffs=[[1, 0], [0, 1]],
    coeffs=[
        normals[grp['boundary:free'], 0],
        normals[grp['boundary:free'], 1]
        ],
    phi=radial_basis_function,
    order=polynomial_order
    )

B = expand_rows(B_temp, idx, nodes.shape[0])
B += expand_rows(B_free, grp['ghosts:free'], nodes.shape[0])
B = B.tocsc()
Bsolver = splu(B)

# construct a matrix that maps the temperature at `nodes` to the time
# derivative of the state vector

# `idx` are the indices of the state vector with a non-zero time derivative
# (i.e., the elements of the state vector that are not enforcing the boundary
# conditions).
idx = np.hstack((grp['interior'], grp['boundary:free']))
A = weight_matrix(
    nodes[idx],
    nodes,
    stencil_size,
    [[1, 0], [2, 0], [0, 1], [0, 2]],
    coeffs=[
        conductivity_xdiff(nodes[idx]),
        conductivity(nodes[idx]),
        conductivity_ydiff(nodes[idx]),
        conductivity(nodes[idx])
        ],
    phi=radial_basis_function,
    order=polynomial_order
    )
A = expand_rows(A, idx, nodes.shape[0])
A = A.tocsc()

def state_derivative(t, z):
    return A.dot(Bsolver.solve(z))

soln = solve_ivp(
    fun=state_derivative,
    t_span=[time_evaluations[0], time_evaluations[-1]],
    y0=z_init,
    method='RK45',
    t_eval=time_evaluations
    )

## PLOTTING
# create the interpolation points
xy_grid = np.mgrid[-2.01:2.01:200j, -2.01:2.01:200j]
xy = xy_grid.reshape(2, -1).T

# create an interpolation matrix that maps the state vector to the temperature
# at `xy`. `idx` consists of indices of the state vector that are temperatures.
idx = np.hstack((grp['interior'], grp['boundary:fixed'], grp['boundary:free']))
I = weight_matrix(
    x=xy,
    p=nodes[idx],
    n=stencil_size,
    diffs=[0, 0],
    phi=radial_basis_function,
    order=polynomial_order
    )
I = expand_cols(I, idx, nodes.shape[0])

# create a mask for points in `xy` that are outside of the domain
is_outside = ~contains(xy, vertices, simplices)

fig1, ax1  = plt.subplots()

k = conductivity(xy)
k[is_outside] = np.nan
k = k.reshape(200, 200)
p = ax1.contourf(*xy_grid, k, np.linspace(1, 4, 11), cmap='viridis')

for smp in simplices:
    ax1.plot(vertices[smp, 0], vertices[smp, 1], 'k-', zorder=1)

for i, (k, v) in enumerate(grp.items()):
    ax1.scatter(nodes[v, 0], nodes[v, 1], s=10, c='C%d' % i, label=k, zorder=2)

ax1.set_aspect('equal')
ax1.set_xlim(-2.2, 2.2)
ax1.set_ylim(-2.2, 2.2)
ax1.grid(ls=':', color='k')
ax1.legend()
cbar = fig1.colorbar(p)
cbar.set_label('heat conductivity')
fig1.tight_layout()
plt.savefig('../figures/fd.n.1.png')

fig2 = plt.figure()
# create the update function for the animation. this plots the solution at time
# `time[index]`
def update(index):
    fig2.clear()
    ax2 = fig2.add_subplot(111)

    z = soln.y[:, index]
    temp = I.dot(z)
    temp[is_outside] = np.nan
    temp = temp.reshape((200, 200))

    for s in simplices:
        ax2.plot(vertices[s, 0], vertices[s, 1], 'k-')

    p = ax2.contourf(
        *xy_grid,
        temp,
        np.linspace(-2.0, 2.0, 10),
        cmap='viridis',
        extend='both'
        )

    ax2.scatter(nodes[:, 0], nodes[:, 1], c='k', s=2)
    ax2.set_title('temperature at time: %.2f' % time_evaluations[index])

    ax2.set_xlim(-2.1, 2.1)
    ax2.set_ylim(-2.1, 2.1)
    ax2.grid(ls=':', color='k')
    ax2.set_aspect('equal')
    fig2.colorbar(p)
    fig2.tight_layout()

    return

ani = FuncAnimation(
    fig=fig2,
    func=update,
    frames=range(len(time_evaluations)),
    repeat=True,
    blit=False)

ani.save('../figures/fd.n.2.gif', writer='imagemagick', fps=3)
plt.show()
