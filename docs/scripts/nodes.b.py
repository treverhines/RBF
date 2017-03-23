''' 
This script demonstrates some of the lower level functions used by
*menodes* to generate nodes over a domain. These functions include,
*rbf.halton.halton*, *rbf.geometry.contains*, *rbf.nodes.disperse*, and
*rbf.nodes.snap_to_boundary*.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.nodes import snap_to_boundary,disperse
from rbf.geometry import contains
from rbf.halton import halton

# Define the problem domain with line segments.
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])

N = 500 # total number of nodes

# create N quasi-uniformly distributed nodes over the unit square
nodes = halton(N,2)
# scale the nodes to encompass the domain
nodes *= 2.0
# remove nodes outside of the domain
nodes = nodes[contains(nodes,vert,smp)]
# evenly disperse the nodes over the domain using 100 iterative steps
for i in range(100): nodes = disperse(nodes,vert,smp)
# snap nodes to the boundary 
nodes,smpid = snap_to_boundary(nodes,vert,smp,delta=0.5)
# identify boundary and interior nodes
interior = smpid == -1
boundary = smpid >  -1

fig,ax = plt.subplots(figsize=(6,6))
# plot the domain
for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
ax.plot(nodes[interior,0],nodes[interior,1],'ko')
ax.plot(nodes[boundary,0],nodes[boundary,1],'bo')
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('../figures/nodes.b.png')
plt.show()


