''' 
In this example we solve the Poisson equation over an L-shaped domain with
fixed boundary conditions. We use the RBF-FD method. The RBF-FD method is
preferable over the spectral RBF method because it is scalable and does not
require the user to specify a shape parameter (assuming that we use odd order
polyharmonic splines to generate the weights).
'''
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
from rbf.pde.nodes import poisson_disc_nodes

# Define the problem domain with line segments.
vert = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
                 [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])

spacing = 0.07 # approximate spacing between nodes

n = 25 # stencil size. Increase this will generally improve accuracy

phi = 'phs3' # radial basis function used to compute the weights. Odd
             # order polyharmonic splines (e.g., phs3) have always performed
             # well for me and they do not require the user to tune a shape
             # parameter. Use higher order polyharmonic splines for higher
             # order PDEs.

order = 2 # Order of the added polynomials. This should be at least as
          # large as the order of the PDE being solved (2 in this case). Larger
          # values may improve accuracy

# generate nodes
nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp)) 
N = nodes.shape[0]

# create the components for the "left hand side" matrix. 
A_interior = weight_matrix(
    x=nodes[groups['interior']], 
    p=nodes, 
    n=n,
    diffs=[[2, 0], [0, 2]],
    phi=phi, 
    order=order)
A_boundary = weight_matrix(
    x=nodes[groups['boundary:all']], 
    p=nodes, 
    n=1,
    diffs=[0, 0]) 
# Expand and add the components together
A  = expand_rows(A_interior, groups['interior'], N)
A += expand_rows(A_boundary, groups['boundary:all'], N)
                           
# create "right hand side" vector
d = np.zeros((N,))
d[groups['interior']] = -1.0
d[groups['boundary:all']] = 0.0

# find the solution at the nodes
u_soln = spsolve(A, d) 

# Create a grid for interpolating the solution
xg, yg = np.meshgrid(np.linspace(0.0, 2.02, 100), np.linspace(0.0, 2.02, 100))
points = np.array([xg.flatten(), yg.flatten()]).T                    

# We can use any method of scattered interpolation (e.g.,
# scipy.interpolate.LinearNDInterpolator). Here we repurpose the RBF-FD method
# to do the interpolation with a high order of accuracy
I = weight_matrix(
    x=points, 
    p=nodes, 
    n=n, 
    diffs=[0, 0],
    phi=phi,
    order=order)
u_itp = I.dot(u_soln)

# mask points outside of the domain
u_itp[~contains(points, vert, smp)] = np.nan 
ug = u_itp.reshape((100, 100)) # fold back into a grid

# make a contour plot of the solution
fig, ax = plt.subplots()
p = ax.contourf(xg, yg, ug, np.linspace(-1e-6, 0.16, 9), cmap='viridis')
ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=4)
for s in smp:
  ax.plot(vert[s, 0], vert[s, 1], 'k-', lw=2)

ax.set_aspect('equal')
ax.set_xlim(-0.05, 2.05)
ax.set_ylim(-0.05, 2.05)
fig.colorbar(p, ax=ax)
fig.tight_layout()
plt.savefig('../figures/fd.i.png')
plt.show()
