''' 
In this example we solve the Poisson equation over an L-shaped domain
with a mix of free and fixed boundary conditions. We use the RBF-FD
method and demonstrate the use of ghost nodes along the free boundary.
'''
import numpy as np
from rbf.fd import weight_matrix, add_rows
from rbf.basis import phs3
from rbf.geometry import contains
from rbf.nodes import min_energy_nodes
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import LinearNDInterpolator


def series_solution(nodes, n=50):
    '''
    The analytical solution for this example
    '''
    x, y = nodes[:,0] - 1.0, nodes[:,1] - 1.0
    out = (1 - x**2)/2 
    for k in range(1, n+1, 2):
        out_k  = 16/np.pi**3
        out_k *= np.sin(k*np.pi*(1 + x)/2)/(k**3*np.sinh(k*np.pi))
        out_k *= (np.sinh(k*np.pi*(1 + y)/2) + 
                  np.sinh(k*np.pi*(1 - y)/2))
        out -= out_k

    return out
    

# Define the problem domain with line segments.
vert = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
                 [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
smp = np.array([[0, 1], [1, 2], [4, 5], [5, 0], [2, 3], [3, 4]])
# define which simplices make up the free and fixed boundary
# conditions
boundary_groups = {'fixed': [0, 1, 2, 3],
                   'free': [4, 5]}

N_nominal = 500 # The nominal number of nodes. The actual number of
                # nodes will be larger in this example

n = 20 # stencil size. Increasing this will generally improve accuracy

basis = phs3 # radial basis function used to compute the weights. Odd
             # order polyharmonic splines (e.g., phs3) have always
             # performed well for me and they do not require the user
             # to tune a shape parameter. Use higher order
             # polyharmonic splines for higher order PDEs.

order = 2 # Order of the added polynomials. This should be at least as
          # large as the order of the PDE being solved (2 in this
          # case). Larger values may improve accuracy

# generate nodes
nodes, groups, normals = min_energy_nodes(
  N_nominal, 
  vert, 
  smp, 
  boundary_groups=boundary_groups,
  boundary_groups_with_ghosts=['free'],
  include_vertices=True) 

N = nodes.shape[0] 

# create the "left hand side" matrix. 
# create the component which evaluates the PDE
A_interior = weight_matrix(nodes[groups['interior']], 
                           nodes,
                           n=n, 
                           diffs=[[2, 0], [0, 2]], 
                           basis=basis, order=order)

# use the ghost nodes to evaluate the PDE at the free boundary nodes
A_ghost = weight_matrix(nodes[groups['boundary:free']], 
                        nodes, 
                        n=n,
                        diffs=[[2, 0], [0, 2]],
                        basis=basis, order=order)

# create the component for the fixed boundary conditions. This is
# essentially an identity operation and so we only need a stencil size
# of 1
A_fixed = weight_matrix(nodes[groups['boundary:fixed']], 
                        nodes, 
                        n=1,
                        diffs=[0, 0]) 

# create the component for the free boundary conditions. This dots the
# derivative with respect to x and y with the x and y components of
# normal vectors on the free surface (i.e., n_x * du/dx + n_y * du/dy)
A_free = weight_matrix(nodes[groups['boundary:free']], 
                       nodes, 
                       n=n,
                       diffs=[[1, 0], [0, 1]],
                       coeffs=[normals[groups['boundary:free'], 0],
                               normals[groups['boundary:free'], 1]],
                       basis=basis, order=order)
                           
# Add the components to the corresponding rows of `A`
A = csc_matrix((N, N))
A = add_rows(A, A_interior, groups['interior'])
A = add_rows(A, A_ghost, groups['ghosts:free'])
A = add_rows(A, A_fixed, groups['boundary:fixed'])
A = add_rows(A, A_free, groups['boundary:free'])
                           
# create "right hand side" vector
d = np.zeros((N,))
d[groups['interior']] = -1.0
d[groups['ghosts:free']] = -1.0
d[groups['boundary:fixed']] = 0.0
d[groups['boundary:free']] = 0.0

# find the solution at the nodes
u_soln = spsolve(A, d) 
error = np.abs(u_soln - series_solution(nodes))

## PLOT THE NUMERICAL SOLUTION AND ITS ERROR
fig, axs = plt.subplots(2, figsize=(6, 8))
# interpolate the solution on a grid
xg, yg = np.meshgrid(np.linspace(-0.05, 2.05, 400), 
                     np.linspace(-0.05, 2.05, 400))
points = np.array([xg.flatten(), yg.flatten()]).T  
u_itp = LinearNDInterpolator(nodes, u_soln)(points)
# mask points outside of the domain
u_itp[~contains(points, vert, smp)] = np.nan 
ug = u_itp.reshape((400, 400)) # fold back into a grid
# make a contour plot of the solution
p = axs[0].contourf(xg, yg, ug, np.linspace(-1e-6, 0.3, 9), cmap='viridis')
fig.colorbar(p, ax=axs[0])
# plot the domain
for s in smp:
  axs[0].plot(vert[s, 0], vert[s, 1], 'k-', lw=2)

# show the locations of the nodes
for i, (k, v)  in enumerate(groups.items()):
    axs[0].plot(nodes[v, 0], nodes[v, 1], 'C%so' % i, 
                markersize=4, label=k)

axs[0].set_title('RBF-FD solution')
axs[0].set_aspect('equal')
axs[0].legend()

# plot the error at the location of the non-ghost nodes 
idx_no_ghosts = np.hstack((groups['interior'], 
                           groups['boundary:free'],
                           groups['boundary:fixed']))
p = axs[1].scatter(nodes[idx_no_ghosts,0], 
                   nodes[idx_no_ghosts,1], 
                   s=20, c=error[idx_no_ghosts])
fig.colorbar(p, ax=axs[1])
# make the background black so its easier to see the colors
axs[1].set_facecolor('k')
axs[1].set_title('Error')
axs[1].set_aspect('equal')

fig.tight_layout()

plt.savefig('../figures/fd.j.png')
plt.show()

