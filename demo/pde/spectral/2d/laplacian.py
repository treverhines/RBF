#!/usr/bin/env python
#
# demonstrate using the spectral RBF method for solving the following 
# PDE:
#
#  (d^2/dx^2 + d^2/dy^2)u(x,y) = f(x,y)  for R < 1.0
#                       u(x,y) = 0.0     for R = 1.0
#
#  R = sqrt(x**2 + y**2)

import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
from rbf.nodes import menodes
from matplotlib import cm
import sympy
import rbf.domain
# set default cmap to viridis if you have it
if 'viridis' in vars(cm):
  plt.rcParams['image.cmap'] = 'viridis'

# total number of nodes
N = 100
basis = rbf.basis.phs3


# symbolic definition of the solution
x,y = sympy.symbols('x,y')
r = sympy.sqrt(x**2 + y**2)
true_soln_sym = (1-r)*sympy.sin(x)*sympy.cos(y)
# numerical solution
true_soln = sympy.lambdify((x,y),true_soln_sym,'numpy')

# symbolic forcing term
forcing_sym = true_soln_sym.diff(x,x) + true_soln_sym.diff(y,y)
# numerical forcing term
forcing = sympy.lambdify((x,y),forcing_sym,'numpy')

# define a circular domain
vert,smp = rbf.domain.circle()

nodes,smpid = menodes(N,vert,smp)
# smpid describes which boundary simplex, if any, the nodes are 
# attached to. If it is -1, then the node is in the interior
boundary, = (smpid>=0).nonzero()
interior, = (smpid==-1).nonzero()

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes
A = np.zeros((N,N))
A[interior]  = basis(nodes[interior],nodes,diff=[2,0]) 
A[interior] += basis(nodes[interior],nodes,diff=[0,2]) 
A[boundary,:] = basis(nodes[boundary],nodes)

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = np.zeros(N)
d[interior] = forcing(nodes[interior,0],nodes[interior,1]) 
d[boundary] = true_soln(nodes[boundary,0],nodes[boundary,1]) 

# find the RBF coefficients that solve the PDE
coeff = np.linalg.solve(A,d)

# create a collection of interpolation points to evaluate the 
# solution. It is easiest to just call menodes again
itp,dummy = menodes(10000,vert,smp,itr=0)

# solution at the interpolation points
soln = basis(itp,nodes).dot(coeff)

# plot the results
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].set_title('RBF solution')
p = ax[0].tripcolor(itp[:,0],itp[:,1],soln)
ax[0].plot(nodes[:,0],nodes[:,1],'ko')
# plot the boundary
for s in smp:
  ax[0].plot(vert[s,0],vert[s,1],'k-',lw=2)

fig.colorbar(p,ax=ax[0])

ax[1].set_title('error')
p = ax[1].tripcolor(itp[:,0],itp[:,1],soln - true_soln(itp[:,0],itp[:,1]))
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
plt.savefig('figures/demo_spectral_laplacian.png')
plt.show()
