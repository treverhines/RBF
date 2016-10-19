#!/usr/bin/env python
#
# Quick demonstration on using the spectral RBF method to solve the 
# Poisson equation

import numpy as np
from rbf.basis import phs3
from rbf.domain import circle
from rbf.nodes import menodes
import matplotlib.pyplot as plt

def forcing(x,y):
  # an arbitrary forcing function
  return -1 + np.sqrt(x**2 + y**2)
  
vert,smp = circle() # use predefined domain geometry
N = 100 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
boundary, = (smpid>=0).nonzero() # identify boundary nodes
interior, = (smpid==-1).nonzero() # identify interior nodes

# create left-hand-side matrix and right-hand-side vector
A = np.empty((N,N))
A[interior]  = phs3(nodes[interior],nodes,diff=[2,0]) 
A[interior] += phs3(nodes[interior],nodes,diff=[0,2]) 
A[boundary,:] = phs3(nodes[boundary],nodes)
d = np.empty(N)
d[interior] = forcing(nodes[interior,0],nodes[interior,1]) 
d[boundary] = 0.0

# solve the PDE
coeff = np.linalg.solve(A,d) # solve for the RBF coefficients
itp = menodes(2000,vert,smp)[0] # evaluate the solution at these points
soln = phs3(itp,nodes).dot(coeff) 

# solution at the interpolation points
plt.tripcolor(itp[:,0],itp[:,1],soln)
plt.plot(nodes[:,0],nodes[:,1],'ko')
plt.colorbar()
plt.show()
