#!/usr/bin/env python
import numpy as np
from rbf.nodes import make_nodes
from PIL import Image
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

### Example 1 
#####################################################################
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

# number of iterations for the node generation algorithm
itr = 100

# step size scaling factor. default is 0.1. smaller values create more 
# uniform spacing after sufficiently many iterations
delta = 0.1

# generate nodes. nodese is a (N,2) array and smpid is a (N,) 
# identifying the simplex, if any, that each node is attached to
nodes1,smpid1 = make_nodes(N,vert,smp,itr=itr,delta=delta)

### Example 2 
#####################################################################
# make node distribution with density defined by the gray-scale Lenna 
# image

# define more complicated domain
t = np.linspace(0,2*np.pi,201)
t = t[:-1]
radius = 0.45*(0.1*np.sin(10*t) + 1.0)
vert = np.array([0.5+radius*np.cos(t),
                 0.5+radius*np.sin(t)]).T
smp = np.array([np.arange(200),np.roll(np.arange(200),-1)]).T
                 
N = 30000
itr = 20
delta = 0.1

# make gray scale image
img = Image.open('Lenna.png')
imga = np.array(img,dtype=float)/256.0
gray = np.linalg.norm(imga,axis=-1)
# normalize so that the max value is 1
gray = gray/gray.max()

# define the node density function
def rho(p):
  # x and y are mapped to integers between 0 and 512
  p = p*512
  p = np.array(p,dtype=int)
  return 1.0001 - gray[511-p[:,1],p[:,0]]


nodes2,smpid2 = make_nodes(N,vert,smp,rho=rho,itr=itr,delta=delta)

### plot results
#####################################################################
fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes1[smpid1==-1,0],nodes1[smpid1==-1,1],'ko')
# plot boundary nodes
ax.plot(nodes1[smpid1>=0,0],nodes1[smpid1>=0,1],'bo')
ax.set_aspect('equal')
ax.set_xlim((-0.1,1.1))
ax.set_ylim((-0.1,1.1))
fig.tight_layout()
plt.savefig('figures/demo_nodes_1.png')

fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes2[smpid2==-1,0],nodes2[smpid2==-1,1],'k.',markersize=2.0)
# plot boundary nodes
ax.plot(nodes2[smpid2>=0,0],nodes2[smpid2>=0,1],'b.',markersize=2.0)
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('figures/demo_nodes_2.png')

plt.show()
