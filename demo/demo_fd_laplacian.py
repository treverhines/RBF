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
logging.basicConfig(level=logging.DEBUG)
# set default cmap to viridis if you have it
if 'viridis' in vars(cm):
  plt.rcParams['image.cmap'] = 'viridis'

def true_soln(pnts):
  # true solution with has zeros on the unit circle
  r = np.sqrt(pnts[:,0]**2 + pnts[:,1]**2)
  soln = (1 - r)*np.sin(pnts[:,0])*np.cos(pnts[:,1])
  return soln 

def forcing(pnts):
  # laplacian of the true solution (forcing term)
  x = pnts[:,0]
  y = pnts[:,1]
  out = ((2*x**2*np.sin(x)*np.cos(y) - 
          2*x*np.cos(x)*np.cos(y) + 
          2*y**2*np.sin(x)*np.cos(y) + 
          2*y*np.sin(x)*np.sin(y) - 
          2*np.sqrt(x**2 + y**2)*np.sin(x)*np.cos(y) - 
          np.sin(x)*np.cos(y))/np.sqrt(x**2 + y**2))

  return out

# stencil size
#S = 20
S = None
# The RBF-FD method allows for polynomial precision, where polynomials 
# of a given order are added to the interpolant used to derive the 
# finite difference weights.  Setting P to 2 adds a second order 
# polynomial to the interpolant.
P = None

# define a circular domain
t = np.linspace(0.0,2*np.pi,100)
vert = np.array([np.cos(t),np.sin(t)]).T
smp = np.array([np.arange(100),np.roll(np.arange(100),-1)]).T

# create the nodes
N = 1000
nodes,smpid = make_nodes(N,vert,smp)
boundary, = (smpid>=0).nonzero()

# basis function used to solve this PDE
basis = rbf.basis.phs3

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes
A  = weight_matrix(nodes,nodes,(2,0),N=S,order=P,basis=basis).toarray() 
A += weight_matrix(nodes,nodes,(0,2),N=S,order=P,basis=basis).toarray() 
A[boundary,:] = weight_matrix(nodes[boundary],nodes,(0,0),N=S,order=P,basis=basis).toarray()

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = forcing(nodes) 
d[boundary] = 0.0 

# find the solution at the nodes
u = np.linalg.solve(A,d)

# create a collection of interpolation points to evaluate the 
# solution. It is easiest to just call make_nodes again 
itp,dummy = make_nodes(10000,vert,smp,itr=0)
itp = nodes
uitp = u
# solution at the interpolation points
#uitp = weight_matrix(itp,nodes,(0,0),N=9,order=2,basis=basis).dot(u)

# plot the results
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_title('RBF solution')
p = ax[0].tripcolor(itp[:,0],itp[:,1],uitp)
ax[0].plot(nodes[:,0],nodes[:,1],'ko')
# plot the boundary
for s in smp:
  ax[0].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[0])

ax[1].set_title('error')
ax[1].plot(nodes[:,0],nodes[:,1],'ko')
p = ax[1].tripcolor(itp[:,0],itp[:,1],uitp - true_soln(itp))
#p = ax[1].tripcolor(nodes[:,0],nodes[:,1],u - true_soln(nodes))
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
