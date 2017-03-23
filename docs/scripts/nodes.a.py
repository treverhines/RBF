''' 
This script demonstrates how to use *menodes* to generate nodes over a
domain with spatially variable density. These nodes can then be used
to solve partial differential equation for this domain using the
spectral RBF method or thr RBF-FD method.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.nodes import menodes

# Define the problem domain with line segments.
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])

# define a node density function. It takes an (N,D) array of positions
# and returns an (N,) array of normalized densities between 0 and 1
def rho(x):
  r = np.sqrt((x[:,0] - 1.0)**2 + (x[:,1] - 1.0)**2)
  return 0.1 + 0.9/((r/0.25)**4 + 1.0)

N = 1000 # total number of nodes

nodes,smpid = menodes(N,vert,smp,rho=rho)
# identify boundary and interior nodes
interior = smpid == -1
boundary = smpid >  -1

fig,ax = plt.subplots(figsize=(6,6))
# plot the domain
for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
ax.plot(nodes[interior,0],nodes[interior,1],'k.')
ax.plot(nodes[boundary,0],nodes[boundary,1],'b.')
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('../figures/nodes.a.png')
plt.show()


