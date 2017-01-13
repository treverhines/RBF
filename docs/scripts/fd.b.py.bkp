''' 
This script demonstrates using the RBF-FD method to calculate 
deformation of an elastic material under the force of gravity. NOTE! I 
have not validated this solution and the code may contain bugs from 
improperly setting up the PDE.
'''
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import rbf
from rbf.fd import weight_matrix

# define the vertices of the problem domain 
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],[1.25,1.0],
                 [1.0,1.25],[0.75,1.0],[0.0,1.0]])
# define the connectivity of the vertices
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,0]])
# number of nodes 
N = 20000
# size of RBF-FD stencils
n = 20
# generate nodes
nodes,smpid = rbf.nodes.menodes(N,vert,smp)
interior, = np.nonzero(smpid == -1) 
# fix the bottom and sides of the domain while keeping the surface 
# free
fix_boundary, = np.nonzero((smpid == 0) | 
                          (smpid == 1) | 
                          (smpid == 6))
free_boundary, = np.nonzero((smpid != 0) & 
                            (smpid != 1) & 
                            (smpid != 6) & 
                            (smpid != -1))
# write out the 2-D equations of motion. This is the most difficult 
# part of setting up the problem and there is no way around it.

## Enforce PDE on interior nodes
#####################################################################
lamb = 1.0 # lame parameters
mu = 1.0
# x component of force resulting from displacement in the x direction
coeffs_xx = [lamb+2*mu,mu]
diffs_xx = [(2,0),(0,2)]
# x component of force resulting from displacement in the y direction. 
coeffs_xy = [lamb,mu]
diffs_xy = [(1,1),(1,1)]
# y component of force resulting from displacement in the x direction.
coeffs_yx = [mu,lamb]
diffs_yx = [(1,1),(1,1)]
# y component of force resulting from displacement in the y direction
coeffs_yy = [lamb+2*mu, mu]
diffs_yy =  [(0,2),(2,0)]
# make the differentiation matrices that enforce the PDE on the 
# interior nodes.
D_xx = weight_matrix(nodes[interior],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
D_xy = weight_matrix(nodes[interior],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
D_yx = weight_matrix(nodes[interior],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
D_yy = weight_matrix(nodes[interior],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
# stack them together
D_x = scipy.sparse.hstack((D_xx,D_xy))
D_y = scipy.sparse.hstack((D_yx,D_yy))
D = scipy.sparse.vstack((D_x,D_y))

## Enforce fixed boundary conditions on bottom and side nodes
#####################################################################
# Enforce that x and y are as specified with the boundary condition. 
# These matrices turn out to be identity matrices, but I include this 
# computation for consistency with the rest of the code
coeffs_xx = [1.0]
diffs_xx = [(0,0)]
coeffs_xy = [0.0]
diffs_xy = [(0,0)]
coeffs_yx = [0.0]
diffs_yx = [(0,0)]
coeffs_yy = [1.0]
diffs_yy = [(0,0)]
dD_fix_xx = weight_matrix(nodes[fix_boundary],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
dD_fix_xy = weight_matrix(nodes[fix_boundary],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
dD_fix_yx = weight_matrix(nodes[fix_boundary],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
dD_fix_yy = weight_matrix(nodes[fix_boundary],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
# stack them together
dD_fix_x = scipy.sparse.hstack((dD_fix_xx,dD_fix_xy))
dD_fix_y = scipy.sparse.hstack((dD_fix_yx,dD_fix_yy))
dD_fix = scipy.sparse.vstack((dD_fix_x,dD_fix_y))

## Enforce free surface boundary conditions on top nodes
#####################################################################
# find the normal vector for each simplex
simplex_normals = rbf.geometry.simplex_outward_normals(vert,smp)
# find the normal vectors for each freeboundary node
normals = simplex_normals[smpid[free_boundary]]
# x component of traction force resulting from x displacement 
coeffs_xx = [normals[:,0]*(lamb+2*mu),normals[:,1]*mu]
diffs_xx = [(1,0),(0,1)]
# x component of traction force resulting from y displacement
coeffs_xy = [normals[:,0]*lamb,normals[:,1]*mu]
diffs_xy = [(0,1),(1,0)]
# y component of traction force resulting from x displacement
coeffs_yx = [normals[:,0]*mu,normals[:,1]*lamb]
diffs_yx = [(0,1),(1,0)]
# y component of force resulting from displacement in the y direction
coeffs_yy = [normals[:,1]*(lamb+2*mu),normals[:,0]*mu]
diffs_yy =  [(0,1),(1,0)]
# make the differentiation matrices that enforce the free surface boundary 
# conditions.
dD_free_xx = weight_matrix(nodes[free_boundary],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
dD_free_xy = weight_matrix(nodes[free_boundary],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
dD_free_yx = weight_matrix(nodes[free_boundary],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
dD_free_yy = weight_matrix(nodes[free_boundary],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
# stack them together
dD_free_x = scipy.sparse.hstack((dD_free_xx,dD_free_xy))
dD_free_y = scipy.sparse.hstack((dD_free_yx,dD_free_yy))
dD_free = scipy.sparse.vstack((dD_free_x,dD_free_y))
# "left hand side" matrix
G = scipy.sparse.vstack((D,dD_fix,dD_free))
G = G.tocsc() # set to csc sparse matrix for efficiency purposes
# x component of body force on interior nodes
f_x = np.zeros(len(interior))
# y component of body force on interior nodes (this where we set gravity)
f_y = np.ones(len(interior))
# x component of fixed boundary condition 
fix_x = np.zeros(len(fix_boundary))
# y component of fixed boundary condition 
fix_y = np.zeros(len(fix_boundary))
# x component of free boundary condition 
free_x = np.zeros(len(free_boundary))
# y component of free boundary condition 
free_y = np.zeros(len(free_boundary))
# "right hand side" vector
d = np.hstack((f_x,f_y,fix_x,fix_y,free_x,free_y))
# find the solution
m = scipy.sparse.linalg.spsolve(G,d)
m_x = m[:N]
m_y = m[N:]
print(m.min())
print(m.max())
plt.quiver(nodes[:,0],nodes[:,1],m_x,m_y)
plt.show()                    


