#!/usr/bin/env python
# This script demonstrates the effect of the bound_force argument in 
# make_nodes
import numpy as np
from rbf.nodes import make_nodes
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.DEBUG)

# define the vertices and simplices for cut annulus
t = np.linspace(0,2*np.pi,100)[:-1]
vert_outer = np.array([2*np.cos(t),2*np.sin(t)]).T
vert_inner = np.array([np.cos(t[::-1]),np.sin(t[::-1])]).T
vert = np.vstack((vert_outer,vert_inner))
smp = np.array([np.arange(198),np.roll(np.arange(198),-1)]).T

# number of nodes
N = 500

# number of iterations for the node generation algorithm
itr = 100

# step size scaling factor. default is 0.1. smaller values create more 
# uniform spacing after sufficiently many iterations
delta = 0.1

# setting bound_force=True ensures that the edges where the annulus is 
# cut will have an appropriate number of boundary nodes. This also 
# makes the function considerably slower
nodes,smpid = make_nodes(N,vert,smp,itr=itr,delta=delta,bound_force=True)

plt.plot(nodes[smpid>=0,0],nodes[smpid>=0,1],'bo')
plt.plot(nodes[smpid==-1,0],nodes[smpid==-1,1],'ko')
plt.show()
