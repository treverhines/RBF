#!/usr/bin/env python
# demonstrate using the spectral RBF method for solving the Laplacian 
# on a 2D domain
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
from rbf.nodes import make_nodes
from matplotlib import cm
import time
import logging
logging.basicConfig(level=logging.DEBUG)
# set default cmap to viridis if you have it
if 'viridis' in vars(cm):
  plt.rcParams['image.cmap'] = 'viridis'


# define a circular domain
B = 50
t = np.linspace(0.0,2*np.pi,B)
vert = np.array([np.cos(t),np.sin(t)]).T
smp = np.array([np.arange(B),np.roll(np.arange(B),-1)]).T

N = 100
nodes,smpid = make_nodes(N,vert,smp,itr=100,delta=0.1)
bnd = np.nonzero(smpid>=0)[0]
A  = rbf.basis.phs3(nodes,nodes,diff=(2,0)) 
A += rbf.basis.phs3(nodes,nodes,diff=(0,2)) 
A[bnd,:] = rbf.basis.phs3(nodes[bnd],nodes)
d = -np.ones(N)
d[smpid>=0] = 0.0
coeff = np.linalg.solve(A,d)

a = time.time()
nodes_itp,dummy = make_nodes(100000,vert,smp,itr=10,n=10,orient=False,sort_nodes=False)
print(1000*(time.time() - a))
soln = rbf.basis.phs3(nodes_itp,nodes).dot(coeff)

fig,ax = plt.subplots()
p = ax.tripcolor(nodes_itp[:,0],nodes_itp[:,1],soln)
fig.colorbar(p,ax=ax)
#ax.plot(nodes[:,0],nodes[:,1],'o')
#ax.set_aspect('equal')
plt.show()
