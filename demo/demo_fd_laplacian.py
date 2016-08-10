#!/usr/bin/env python
#
# demonstrate using the RBF-FD method for solving the following 
# PDE:
#
#  (d^2/dx^2 + d^2/dy^2)u(x,y) = f(x,y)  for R < 1.0
#                       u(x,y) = 0.0     for R = 1.0
#
#  R = sqrt(x**2 + y**2)
#
# This script is intended to be nearly identical to 
# demo_spectral_laplacian.py so that the user can easily compare the 
# spectral and finite difference methods. It is NOT intended to be a 
# demonstration of how to efficiently use the RBF-FD method. See 
# demo_fd_annulus.py for a demonstration of how to efficiently solve a 
# PDE with the RBF-FD method
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
from rbf.nodes import make_nodes
from rbf.fd import weight_matrix
from matplotlib import cm
import logging
import rbf.domain
import scipy.sparse
import sympy
logging.basicConfig(level=logging.DEBUG)
# set default cmap to viridis if you have it
if 'viridis' in vars(cm):
  plt.rcParams['image.cmap'] = 'viridis'


x,y = sympy.symbols('x,y')
r = sympy.sqrt(x**2 + y**2)
true_soln_sym = (1-r)*sympy.sin(x)*sympy.cos(y)
true_soln = sympy.lambdify((x,y),true_soln_sym,'numpy')

forcing_sym = true_soln_sym.diff(x,x) + true_soln_sym.diff(y,y)
forcing = sympy.lambdify((x,y),forcing_sym,'numpy')

# define a circular domain
vert,smp = rbf.domain.circle()

# create the nodes
N = 1000
S = None
nodes,smpid = make_nodes(N,vert,smp)
# smpid describes which boundary simplex, if any, the nodes are 
# attached to. If it is -1, then the node is in the interior
boundary, = (smpid>=0).nonzero()
interior, = (smpid==-1).nonzero()

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes. The third argument to weight_matrix 
# describes the derivates order for each spatial dimension
A = scipy.sparse.lil_matrix((N,N))
A[interior,:]  = weight_matrix(nodes[interior],nodes,[2,0])
A[interior,:] += weight_matrix(nodes[interior],nodes,[0,2])
A[boundary,:]  = weight_matrix(nodes[boundary],nodes,[0,0])
# convert A to a csr matrix for efficient solving
A = A.tocsr()

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = np.zeros(N)
d[interior] = forcing(nodes[interior,0],nodes[interior,1]) 
d[boundary] = true_soln(nodes[boundary,0],nodes[boundary,1])

# find the solution at the nodes
u = scipy.sparse.linalg.spsolve(A,d)

# plot the results
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_title('RBF solution')
p = ax[0].tripcolor(nodes[:,0],nodes[:,1],u)
ax[0].plot(nodes[:,0],nodes[:,1],'ko')
# plot the boundary
for s in smp:
  ax[0].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[0])

ax[1].set_title('error')
ax[1].plot(nodes[:,0],nodes[:,1],'ko')

p = ax[1].tripcolor(nodes[:,0],nodes[:,1],u - true_soln(nodes[:,0],nodes[:,1]))
for s in smp:
  ax[1].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[1])
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlim((-1.05,1.05))
ax[0].set_ylim((-1.05,1.05))
ax[1].set_xlim((-1.05,1.05))
ax[1].set_ylim((-1.05,1.05))
fig.tight_layout()
plt.savefig('figures/demo_fd_laplacian.png')
plt.show()
