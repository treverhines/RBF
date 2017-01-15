#!/usr/bin/env python
import numpy as np
from rbf.nodes import menodes
from PIL import Image
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

# simple node generation over a square
# define the domain
vert = [[0.0,0.0],
        [1.0,0.0],
        [1.0,1.0],
        [0.0,1.0]]
smp = [[0,1],
       [1,2],
       [2,3],
       [3,0]]

# number of nodes
N = 1000

# generate nodes. nodese is a (N,2) array and smpid is a (N,) 
# identifying the simplex, if any, that each node is attached to
nodes,smpid = menodes(N,vert,smp)

boundary, = np.nonzero(smpid>=0)
interior, = np.nonzero(smpid==-1)

### plot results
#####################################################################
fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes[interior,0],nodes[interior,1],'ko')
# plot boundary nodes
ax.plot(nodes[boundary,0],nodes[boundary,1],'bo')
ax.set_aspect('equal')
ax.set_xlim((-0.1,1.1))
ax.set_ylim((-0.1,1.1))
fig.tight_layout()
plt.savefig('figures/square.png')
plt.show()
