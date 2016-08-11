#!/usr/bin/env python
import numpy as np
from rbf.nodes import make_nodes
from PIL import Image
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

# make node distribution with density defined by the gray-scale Lenna 
# image

# define the domain
t = np.linspace(0,2*np.pi,201)
t = t[:-1]
radius = 0.45*(0.1*np.sin(10*t) + 1.0)
vert = np.array([0.5+radius*np.cos(t),0.5+radius*np.sin(t)]).T
smp = np.array([np.arange(200),np.roll(np.arange(200),-1)]).T
                 
N = 30000

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


nodes,smpid = make_nodes(N,vert,smp,rho=rho)
interior, = np.nonzero(smpid==-1)
boundary, = np.nonzero(smpid>=0)

### plot results
#####################################################################
fig,ax = plt.subplots()
# plot interior nodes
ax.plot(nodes[interior,0],nodes[interior,1],'k.',markersize=2)
# plot boundary nodes
ax.plot(nodes[boundary,0],nodes[boundary,1],'b.',markersize=2)
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig('figures/lenna.png')
plt.show()
