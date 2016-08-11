#!/usr/bin/env python
#
# demonstrate using the RBF-FD method for solving the following 
# PDE:
#
#  (d^2/dx^2 + d^2/dy^2 + d^2/dz)u(x,y,z) = f(x,y,z)  for R < 1.0
#                                u(x,y,z) = 0.0       for R = 1.0
#
#  R = sqrt(x**2 + y**2 + z**2)
#
# The true solution (and corresponding f) are defined in the script 
# and can be easily modified

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
from scipy.interpolate import griddata
import numpy as np
from mayavi import mlab 
logging.basicConfig(level=logging.DEBUG)

def scatter_contour(nodes,vals,**kwargs):
  pts = 100j
  xmin,xmax = np.min(nodes[:,0]),np.max(nodes[:,0])
  ymin,ymax = np.min(nodes[:,1]),np.max(nodes[:,1])
  zmin,zmax = np.min(nodes[:,2]),np.max(nodes[:,2])
  x,y,z = np.mgrid[xmin:xmax:pts, ymin:ymax:pts, zmin:zmax:pts]
  f = griddata(nodes, vals, (x,y,z),method='linear')
  p = mlab.contour3d(x,y,z,f,**kwargs)
  mlab.colorbar(p)


# total number of nodes
N = 5000
P = 2

# symbolic definition of the solution
x,y,z = sympy.symbols('x,y,z')
r = sympy.sqrt(x**2 + y**2 + z**2)
true_soln_sym = (1-r)*sympy.sin(x)*sympy.cos(y)*sympy.cos(z)
# numerical solution
true_soln = sympy.lambdify((x,y,z),true_soln_sym,'numpy')

# symbolic forcing term
forcing_sym = true_soln_sym.diff(x,x) + true_soln_sym.diff(y,y) + true_soln_sym.diff(z,z)
# numerical forcing term
forcing = sympy.lambdify((x,y,z),forcing_sym,'numpy')

# smpid describes which boundary simplex, if any, the nodes are 
# attached to. If it is -1, then the node is in the interior
vert,smp = rbf.domain.sphere()
nodes,smpid = make_nodes(N,vert,smp,itr=10)
interior, = np.nonzero(smpid==-1)
boundary, = np.nonzero(smpid>=0)

# create the left-hand-side matrix which is the Laplacian of the basis 
# function for interior nodes and the undifferentiated basis functions 
# for the boundary nodes. The third argument to weight_matrix 
# describes the derivates order for each spatial dimension
A = scipy.sparse.lil_matrix((N,N))
A[interior,:]  = weight_matrix(nodes[interior],nodes,[[2,0,0],[0,2,0],[0,0,2]],order=P)
A[boundary,:]  = weight_matrix(nodes[boundary],nodes,[0,0,0],order=P)
# convert A to a csr matrix for efficient solving
A = A.tocsr()

# create the right-hand-side vector, consisting of the forcing term 
# for the interior nodes and zeros for the boundary nodes
d = np.zeros(N)
d[interior] = forcing(nodes[interior,0],nodes[interior,1],nodes[interior,2]) 
d[boundary] = true_soln(nodes[boundary,0],nodes[boundary,1],nodes[boundary,2])

# find the solution at the nodes
u = scipy.sparse.linalg.spsolve(A,d)
utrue = true_soln(nodes[:,0],nodes[:,1],nodes[:,2])
err = u - utrue

mlab.figure(1)
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,color=(1.0,1.0,1.0),opacity=0.1)
scatter_contour(nodes,err,contours=8,opacity=0.3)
mlab.points3d(nodes[:,0],nodes[:,1],nodes[:,2],color=(0.0,0.0,0.0),scale_factor=0.01)
mlab.title('error')
mlab.figure(2)
mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,color=(1.0,1.0,1.0),opacity=0.1)
scatter_contour(nodes,u,contours=8,opacity=0.3)
mlab.points3d(nodes[:,0],nodes[:,1],nodes[:,2],color=(0.0,0.0,0.0),scale_factor=0.01)
mlab.title('solution')
mlab.show()
