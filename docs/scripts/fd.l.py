'''
This script solves the 2-D wave equation on an L-shaped domain with absorbing
boundary conditions.
'''
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.sparse.linalg import splu, LinearOperator, eigs
from scipy.integrate import solve_ivp

from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes
from rbf.pde.geometry import contains

logging.basicConfig(level=logging.DEBUG)

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
# the times where we will evaluate the solution (these are not the time steps)
times = np.linspace(0.0, 4.0, 41)
# the spacing between nodes
spacing = 0.05
# the wave speed
rho = 1.0
# the hyperviscosity factor
nu = 1e-6
# whether to plot the eigenvalues for the state differentiation matrix. All the
# eigenvalues must have a negative real component for the time stepping to be
# stable
plot_eigs = False
# number of nodes used for each RBF-FD stencil
stencil_size = 50
# the polynomial order for generating the RBF-FD weights
order = 4
# the RBF used for generating the RBF-FD weights
phi = 'phs5'

# generate the nodes
nodes, groups, normals = poisson_disc_nodes(
    spacing,
    (vert, smp),
    boundary_groups={'all':range(len(smp))},
    boundary_groups_with_ghosts=['all'])
node_size = nodes.shape[0]

# We will solve the wave equation by numerically integrating the state vector
# `z`. The state vector is a concatenation of `u`, `v`, and `b`. `u` are the
# displacements at the interior and boundary; `v` are the velocities at the
# interior and boundary; and `b` are the boundary conditions.

# create a new node group for convenience
groups['interior+boundary:all'] = np.hstack((groups['interior'],
                                             groups['boundary:all']))

# create the initial displacements at the interior and boundary, the velocities
# are zero
r = np.sqrt((nodes[groups['interior+boundary:all'], 0] - 0.5)**2 +
            (nodes[groups['interior+boundary:all'], 1] - 0.5)**2)

u_init = 1.0/(1 + (r/0.1)**4)
v_init = np.zeros_like(u_init)
b_init = np.zeros(len(groups['boundary:all']))
z_init = np.hstack((u_init, v_init, b_init))
u_size = u_init.shape[0]
v_size = v_init.shape[0]
b_size = b_init.shape[0]
z_size = u_size + v_size + b_size

# construct a matrix that maps the displacements at `nodes` to `u`
U = weight_matrix(
    x=nodes[groups['interior+boundary:all']],
    p=nodes,
    n=1,
    diffs=(0, 0))
U = U.tocsc()    

# construct a matrix that maps the displacements at `nodes` to `b`
B = weight_matrix(
    x=nodes[groups['boundary:all']],
    p=nodes,
    n=stencil_size,
    diffs=[(1, 0), (0, 1)],
    coeffs=[normals[groups['boundary:all'], 0],
            normals[groups['boundary:all'], 1]],
    phi=phi,
    order=order)
B = B.tocsc()
B1 = B[:, groups['interior+boundary:all']]    
B2 = B[:, groups['ghosts:all']]    
B2solver = splu(B2)

# construct a matrix that maps the displacements at `nodes` to the time
# derivative of `v`
D = rho**2 * weight_matrix(
    x=nodes[groups['interior+boundary:all']],
    p=nodes,
    n=stencil_size,
    diffs=[(2, 0), (0, 2)],
    phi=phi,
    order=order)
D = D.tocsc()

# construct a matrix that maps the displacements at `nodes` to the time
# derivative of `b`
C = -rho * weight_matrix(
    x=nodes[groups['boundary:all']],
    p=nodes,
    n=stencil_size,
    diffs=[(2, 0), (0, 2)],
    phi=phi,
    order=order)
C = C.tocsc()

# construct a matrix that maps `v` to the acceleration of `u` due to
# hyperviscosity.
H = -nu*weight_matrix(
    x=nodes[groups['interior+boundary:all']],
    p=nodes[groups['interior+boundary:all']],
    n=stencil_size,
    diffs=[(4, 0), (0, 4)],
    phi='phs5',
    order=4)
H = H.tocsc()

# create a function used for time stepping. this returns the time derivative of
# the state vector
def state_derivative(t, z):
    u = z[:u_size]
    v = z[u_size:(u_size + v_size)]
    b = z[(u_size + v_size):]

    disp = np.zeros((node_size,))
    disp[groups['interior+boundary:all']] = u
    disp[groups['ghosts:all']] = B2solver.solve(b - B1.dot(u))
    
    dudt = v
    dvdt = D.dot(disp) + H.dot(v)
    dbdt = C.dot(disp)

    out = np.hstack((dudt, dvdt, dbdt))
    return out

if plot_eigs:
    L = LinearOperator((z_size, z_size), matvec=lambda x:state_derivative(0.0, x))
    print('computing eigenvectors ...')
    vals = eigs(L, z_size - 2, return_eigenvectors=False)
    print('done')
    print('min real: %s' % np.min(vals.real))
    print('max real: %s' % np.max(vals.real))
    print('min imaginary: %s' % np.min(vals.imag))
    print('max imaginary: %s' % np.max(vals.imag))
    fig, ax = plt.subplots()
    ax.plot(vals.real, vals.imag, 'ko')
    ax.set_title('eigenvalues of the state differentiation matrix')
    ax.set_xlabel('real')
    ax.set_ylabel('imaginary')
    ax.grid(ls=':')

print('performing time integration ...')
soln = solve_ivp(
    fun=state_derivative,
    t_span=[times[0], times[-1]],
    y0=z_init,
    method='RK45',
    t_eval=times)
print('done')

## PLOTTING
# create the interpolation points
xgrid, ygrid = np.meshgrid(np.linspace(0.0, 2.01, 100),
                           np.linspace(0.0, 2.01, 100))
xy = np.array([xgrid.flatten(), ygrid.flatten()]).T

# create an interpolation matrix that maps `u` to the displacements at `xy`
I = weight_matrix(
    x=xy, 
    p=nodes[groups['interior+boundary:all']], 
    n=stencil_size, 
    diffs=(0, 0), 
    phi=phi,
    order=order)

# create a mask for points in `xy` that are outside of the domain
is_outside = ~contains(xy, vert, smp)

fig = plt.figure()

# create the update function for the animation. this plots the solution at time
# `time[index]`
def update(index):
    fig.clear()
    ax = fig.add_subplot(111)

    z = soln.y[:, index]
    u = z[:u_size]
    
    u_xy = I.dot(u)
    u_xy[is_outside] = np.nan 
    u_xy = u_xy.reshape((100, 100))

    for s in smp:
        ax.plot(vert[s, 0], vert[s, 1], 'k-')

    p = ax.contourf(
        xgrid, ygrid, u_xy, 
        np.linspace(-0.4, 0.4, 21), 
        cmap='seismic',
        extend='both')

    ax.scatter(nodes[:, 0], nodes[:, 1], c='k', s=2)
    ax.scatter(nodes[groups['boundary:all'][[0]], 0], 
               nodes[groups['boundary:all'][[0]], 1], c='k', s=10)
    ax.set_title('time: %.2f' % times[index])

    ax.set_xlim(-0.05, 2.05)     
    ax.set_ylim(-0.05, 2.05)     
    ax.grid(ls=':', color='k')   
    ax.set_aspect('equal')
    fig.colorbar(p)
    fig.tight_layout()

    return

ani = FuncAnimation(
    fig=fig, 
    func=update, 
    frames=range(soln.y.shape[1]), 
    repeat=True,
    blit=False)
    
plt.show()
