'''
This script demonstrates using RBF-FD to solve Poisson's equation with mixed
boundary conditions on a domain with a hole inside of it.
'''
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from rbf.pde.nodes import poisson_disc_nodes
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
from rbf.interpolate import KNearestRBFInterpolant

logging.basicConfig(level=logging.DEBUG)

node_spacing = 0.25
radial_basis_function = 'phs3' # 3rd order polyharmonic spline
polynomial_order = 2
stencil_size = 30 

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
    'box': [0, 1, 2, 3],
    'circle': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }
    
# Generate the nodes. `groups` identifies which group each node belongs to.
# `normals` contains the normal vectors corresponding to each node (nan if the
# node is not on a boundary). We are giving the circle boundary nodes ghost
# nodes to improve the accuracy of the free boundary constraint.
nodes, groups, normals = poisson_disc_nodes(
    node_spacing,
    (vertices, simplices),
    boundary_groups=boundary_groups,
    boundary_groups_with_ghosts=['circle']
    )

# Create the LHS and RHS enforcing that the Lapacian is -5 at the interior
# nodes
A_interior = weight_matrix(
    nodes[groups['interior']],
    nodes,
    stencil_size,
    [[2, 0], [0, 2]],
    phi=radial_basis_function,
    order=polynomial_order
    )

b_interior = np.full(len(groups['interior']), -5.0)

# Enforce that the solution is x at the box boundary nodes.
A_boundary_box = weight_matrix(
    nodes[groups['boundary:box']],
    nodes,
    stencil_size,
    [0, 0],
    phi=radial_basis_function,
    order=polynomial_order
    )

b_boundary_box = nodes[groups['boundary:box'], 0]

# Enforce a free boundary at the circle boundary nodes.
A_boundary_circle = weight_matrix(
    nodes[groups['boundary:circle']],
    nodes,
    stencil_size,
    [[1, 0], [0, 1]],
    coeffs=[
        normals[groups['boundary:circle'], 0],
        normals[groups['boundary:circle'], 1]
        ],
    phi=radial_basis_function,
    order=polynomial_order
    )

b_boundary_circle = np.zeros(len(groups['boundary:circle']))

# Use the extra degrees of freedom from the ghost nodes to enforce that the
# Laplacian is -5 at the circle boundary nodes.
A_ghosts_circle = weight_matrix(
    nodes[groups['boundary:circle']],
    nodes,
    stencil_size,
    [[2, 0], [0, 2]],
    phi=radial_basis_function,
    order=polynomial_order
    )

b_ghosts_circle = np.full(len(groups['boundary:circle']), -5.0)

# Combine the LHS and RHS components and solve
A = sp.vstack(
    (A_interior, A_boundary_box, A_boundary_circle, A_ghosts_circle)
    ).tocsc()

b = np.hstack(
    (b_interior, b_boundary_box, b_boundary_circle, b_ghosts_circle)
    )

soln = sp.linalg.spsolve(A, b)

# The rest is just plotting...
fig, ax = plt.subplots()
for smp in simplices:
    ax.plot(vertices[smp, 0], vertices[smp, 1], 'k-')

for name, idx in groups.items():
    ax.plot(nodes[idx, 0], nodes[idx, 1], '.', label=name)

points_grid = np.mgrid[
    nodes[:, 0].min():nodes[:, 0].max():200j,
    nodes[:, 1].min():nodes[:, 1].max():200j,
    ]

points_flat = points_grid.reshape(2, -1).T
soln_interp_flat = KNearestRBFInterpolant(
    nodes, soln,
    phi=radial_basis_function,
    k=stencil_size,
    order=polynomial_order
    )(points_flat)
    
soln_interp_flat[~contains(points_flat, vertices, simplices)] = np.nan
soln_interp_grid = soln_interp_flat.reshape(points_grid.shape[1:])
p = ax.contourf(*points_grid, soln_interp_grid, cmap='jet')
fig.colorbar(p)
ax.legend(loc=2)
ax.grid(ls=':')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.set_title('Interpolated solution')
fig.tight_layout()
plt.savefig('../figures/fd.m.png')
plt.show()
