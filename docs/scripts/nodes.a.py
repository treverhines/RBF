''' 
This script demonstrates how to use `min_energy_nodes` to generate
nodes over a domain with spatially variable density. These nodes can
then be used to solve partial differential equation for this domain
using the spectral RBF method or the RBF-FD method.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.pde.nodes import min_energy_nodes

# Define the problem domain with line segments.
vert = np.array([[2.0, 1.0], [1.0, 1.0], [1.0, 2.0], 
                 [0.0, 2.0], [0.0, 0.0], [2.0, 0.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
# define which simplices belong to which boundary groups
boundary_groups = {'inside_corner': [0, 1],
                   'outside_corner': [2, 3, 4, 5]}
# define which boundary groups get ghost nodes
boundary_groups_with_ghosts = ['outside_corner']
# total number of nodes 
N = 500
# define a node density function. It takes an (N, D) array of positions
# and returns an (N,) array of normalized densities between 0 and 1
def rho(x):
  r = np.sqrt((x[:, 0] - 1.0)**2 + (x[:, 1] - 1.0)**2)
  return 0.2 + 0.8/((r/0.3)**4 + 1.0)

nodes, groups, normals = min_energy_nodes(
    N, 
    (vert, smp),
    rho=rho,
    boundary_groups=boundary_groups,
    boundary_groups_with_ghosts=boundary_groups_with_ghosts,
    include_vertices=True)

# plot the results
fig, ax = plt.subplots(figsize=(6, 6))
# plot the domain
for s in smp: 
  ax.plot(vert[s, 0], vert[s, 1], 'k-')

# plot the different node groups and their normal vectors
for i, (name, idx) in enumerate(groups.items()):
  ax.plot(nodes[idx, 0], nodes[idx, 1], 'C%s.' % i, label=name, ms=8)
  ax.quiver(nodes[idx, 0], nodes[idx, 1], 
            normals[idx, 0], normals[idx, 1], 
            color='C%s' % i)

ax.legend()
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('../figures/nodes.a.png')
plt.show()


