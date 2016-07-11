#!/usr/bin/env python
# This uses the RBF-FD method to solve the Laplacian over a cut 
# annulus
import numpy as np
from rbf.nodes import make_nodes
from rbf.geometry import simplex_outward_normals
from rbf.stencil import nearest
from rbf.fd import weight_matrix
import matplotlib.pyplot as plt
import logging
import scipy.sparse
logging.basicConfig(level=logging.DEBUG)

def make_ghost_nodes(nodes,smpid,idx,vert,smp):
  nodes = np.asarray(nodes)
  smpid = np.asarray(smpid)
  sub_nodes = nodes[idx]
  sub_smpid = smpid[idx]
  if np.any(sub_smpid == -1):
    raise ValueError('cannot make a ghost node for an interior node')
    
  norms = simplex_outward_normals(vert,smp)[sub_smpid]
  dummy,dx = nearest(sub_nodes,nodes,2,vert=vert,smp=smp)
  # distance to the nearest neighbors
  dx = dx[:,[1]]
  ghosts = sub_nodes + dx*norms   
  return ghosts

# define the vertices and simplices for cut annulus
t = np.linspace(0,2*np.pi,100)[:-1]
vert_outer = np.array([2*np.cos(t),2*np.sin(t)]).T
vert_inner = np.array([np.cos(t[::-1]),np.sin(t[::-1])]).T
vert = np.vstack((vert_outer,vert_inner))
smp = np.array([np.arange(198),np.roll(np.arange(198),-1)]).T

S = 10
P = 1
# number of nodes
N = 500

# number of iterations for the node generation algorithm
itr = 50

# step size scaling factor. default is 0.1. smaller values create more 
# uniform spacing after sufficiently many iterations
delta = 0.1

# setting bound_force=True ensures that the edges where the annulus is 
# cut will have an appropriate number of boundary nodes. This also 
# makes the function considerably slower
nodes,smpid = make_nodes(N,vert,smp,itr=itr,delta=delta,bound_force=True)

# identify nodes associated with the different boundary types
cut_top, = (smpid==197).nonzero()
cut_bot, = (smpid==98).nonzero()
# edges of the annulus minus the cut
boundary, = ((smpid>=0) & (smpid!=197) & (smpid!=98)).nonzero()
interior, = (smpid==-1).nonzero()

# add ghost nodes
ghost_nodes = make_ghost_nodes(nodes,smpid,boundary,vert,smp)
nodes = np.vstack((nodes,ghost_nodes))
# ghost node indices
ghost = N + np.arange(len(boundary))

# find boundary normal vectors
normals = simplex_outward_normals(vert,smp)[smpid[boundary]]

# do not build stencils which cross this line
bnd_vert = np.array([[0.0,0.0],[5.0,0.0]])
bnd_smp = np.array([[0,1]])
# build lhs
A_interior = (weight_matrix(nodes[interior],nodes,diff=(2,0),N=S,order=P,vert=bnd_vert,smp=bnd_smp) + 
              weight_matrix(nodes[interior],nodes,diff=(0,2),N=S,order=P,vert=bnd_vert,smp=bnd_smp))

A_ghost = (weight_matrix(nodes[boundary],nodes,diff=(2,0),N=S,order=P,vert=bnd_vert,smp=bnd_smp) + 
           weight_matrix(nodes[boundary],nodes,diff=(0,2),N=S,order=P,vert=bnd_vert,smp=bnd_smp))

n1 = scipy.sparse.diags(normals[:,0],0)
n2 = scipy.sparse.diags(normals[:,1],0)
A_boundary = (n1*weight_matrix(nodes[boundary],nodes,diff=(1,0),N=S,order=P,vert=bnd_vert,smp=bnd_smp) +
              n2*weight_matrix(nodes[boundary],nodes,diff=(0,1),N=S,order=P,vert=bnd_vert,smp=bnd_smp))

A_cut_top = weight_matrix(nodes[cut_top],nodes,diff=(0,0),N=S,order=P,vert=bnd_vert,smp=bnd_smp)
A_cut_bot = weight_matrix(nodes[cut_bot],nodes,diff=(0,0),N=S,order=P,vert=bnd_vert,smp=bnd_smp)

A = scipy.sparse.vstack((A_interior,A_ghost,A_boundary,A_cut_top,A_cut_bot))

# build the rhs in the same order
d_interior = np.zeros(interior.shape[0])
d_ghost = np.zeros(ghost.shape[0])
d_boundary = np.zeros(boundary.shape[0])
d_cut_top = np.ones(cut_top.shape[0])
d_cut_bot = -np.ones(cut_bot.shape[0])

d = np.concatenate((d_interior,d_ghost,d_boundary,d_cut_top,d_cut_bot))

soln = scipy.sparse.linalg.spsolve(A,d)

p = plt.tripcolor(nodes[:,0],nodes[:,1],soln)
for s in smp:
  plt.plot(vert[s,0],vert[s,1],'k-')
  
plt.colorbar(p)

plt.figure(2)
plt.plot(nodes[cut_top,0],nodes[cut_top,1],'ro')
plt.plot(nodes[cut_bot,0],nodes[cut_bot,1],'go')
plt.plot(nodes[boundary,0],nodes[boundary,1],'bo')
plt.plot(nodes[ghost,0],nodes[ghost,1],'ko')
plt.plot(nodes[interior,0],nodes[interior,1],'ko',markersize=5)
plt.show()
