''' 
This script demonstrates using the RBF-FD method to calculate static 
deformation of a two-dimensional elastic material subject to a uniform 
body force such as gravity. The elastic material has a fixed boundary 
condition on one side and the remaining sides have a free surface 
boundary condition.  This script also demonstrates using ghost nodes 
which, for all intents and purposes, are necessary when dealing with 
Neumann boundary conditions.
'''
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rbf.nodes import menodes,neighbors
from rbf.fd import weight_matrix
from rbf.geometry import simplex_outward_normals

def expand_rows(A,idx,rout):
    '''
    Expand the rows of the (rin,c) matrix `A` into an (rout,c) matrix.

    Parameters
    ----------
    A : (rin,c) sparse matrix
    idx: (rin,) int array 
        mapping from the input rows to the output rows
    rout: int
        size of the output array    
    '''
    A = coo_matrix(A)
    idx = np.asarray(idx,dtype=int)
    shape = (rout,A.shape[1])
    out = csc_matrix((A.data,(idx[A.row],A.col)),shape=shape)
    return out


## User defined parameters
#####################################################################
# define the vertices of the problem domain. Note that the first 
# simplex will be fixed, and the others will be free
vert = np.array([[0.0,0.0],[0.0,1.0],[2.0,1.0],[2.0,0.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,0]])
# number of nodes 
N = 1000
# size of RBF-FD stencils
n = 20
# lame parameters
lamb = 1.0
mu = 1.0
# z component of body force
body_force = 1.0

## Build and solve for displacements and strain
#####################################################################
# generate nodes. Read the documentation for *menodes* to tune it and 
# allow for variable node densities.
nodes,smpid = menodes(N,vert,smp)
# find the indices for interior nodes, fixed boundary nodes, and free 
# boundary nodes. This is done by looking at the array *smpid* which 
# tells us the simplex that each boundary node is attached to.
int_idx = np.nonzero(smpid == -1)[0].tolist()
# fix the bottom nodes and keep other boundaries free
fix_idx = np.nonzero(smpid == 0)[0].tolist()
free_idx = np.nonzero(smpid > 0)[0].tolist()
# find the normal vector for each simplex
simplex_normals = simplex_outward_normals(vert,smp)
# find the normal vectors for each free boundary node
normals = simplex_normals[smpid[free_idx]]
# add ghost nodes to greatly improve accuracy at the free surface
dx = np.min(neighbors(nodes,2)[1][:,1])
nodes = np.vstack((nodes,nodes[free_idx] + dx*normals))
## Enforce the PDE on interior nodes AND the free surface nodes 
# x component of force resulting from displacement in the x direction.
coeffs_xx = [lamb+2*mu,mu]
diffs_xx = [(2,0),(0,2)]
# x component of force resulting from displacement in the y direction.
coeffs_xy = [lamb,mu]
diffs_xy = [(1,1),(1,1)]
# y component of force resulting from displacement in the x direction.
coeffs_yx = [mu,lamb]
diffs_yx = [(1,1),(1,1)]
# y component of force resulting from displacement in the y direction.
coeffs_yy = [lamb+2*mu,mu]
diffs_yy =  [(0,2),(2,0)]
# make the differentiation matrices that enforce the PDE on the 
# interior nodes.
D_xx = weight_matrix(nodes[int_idx+free_idx],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
D_xx = expand_rows(Dxx,int_idx+free_idx,N)
D_xy = weight_matrix(nodes[int_idx+free_idx],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
D_yx = weight_matrix(nodes[int_idx+free_idx],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
D_yy = weight_matrix(nodes[int_idx+free_idx],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
# stack them together
D_x = scipy.sparse.hstack((D_xx,D_xy))
D_y = scipy.sparse.hstack((D_yx,D_yy))
D = scipy.sparse.vstack((D_x,D_y))
## Enforce fixed boundary conditions
# Enforce that x and y are as specified with the fixed boundary 
# condition. These matrices turn out to be identity matrices, but I 
# include this computation for consistency with the rest of the code. 
# feel free to comment out the next couple lines and replace it with 
# an appropriately sized sparse identity matrix.
coeffs_xx = [1.0]
diffs_xx = [(0,0)]
coeffs_xy = [0.0]
diffs_xy = [(0,0)]
coeffs_yx = [0.0]
diffs_yx = [(0,0)]
coeffs_yy = [1.0]
diffs_yy = [(0,0)]
dD_fix_xx = weight_matrix(nodes[fix_idx],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
dD_fix_xy = weight_matrix(nodes[fix_idx],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
dD_fix_yx = weight_matrix(nodes[fix_idx],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
dD_fix_yy = weight_matrix(nodes[fix_idx],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
dD_fix_x = scipy.sparse.hstack((dD_fix_xx,dD_fix_xy))
dD_fix_y = scipy.sparse.hstack((dD_fix_yx,dD_fix_yy))
dD_fix = scipy.sparse.vstack((dD_fix_x,dD_fix_y))
## Enforce free surface boundary conditions
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
dD_free_xx = weight_matrix(nodes[free_idx],nodes,diffs_xx,coeffs=coeffs_xx,n=n)
dD_free_xy = weight_matrix(nodes[free_idx],nodes,diffs_xy,coeffs=coeffs_xy,n=n)
dD_free_yx = weight_matrix(nodes[free_idx],nodes,diffs_yx,coeffs=coeffs_yx,n=n)
dD_free_yy = weight_matrix(nodes[free_idx],nodes,diffs_yy,coeffs=coeffs_yy,n=n)
# stack them together
dD_free_x = scipy.sparse.hstack((dD_free_xx,dD_free_xy))
dD_free_y = scipy.sparse.hstack((dD_free_yx,dD_free_yy))
dD_free = scipy.sparse.vstack((dD_free_x,dD_free_y))
## Create the "right hand side" vector components
# body force vector components
f_x = np.zeros_like(int_idx+free_idx)
f_y = body_force*np.ones_like(int_idx+free_idx) # THIS IS WHERE GRAVITY IS ADDED
# fixed boundary conditions
fix_x = np.zeros_like(fix_idx)
fix_y = np.zeros_like(fix_idx)
# free boundary conditions
free_x = np.zeros_like(free_idx)
free_y = np.zeros_like(free_idx)
## Combine and solve
# "left hand side" matrix
G = scipy.sparse.vstack((D,dD_fix,dD_free)).tocsr()
# "right hand side" vector
d = np.hstack((f_x,f_y,fix_x,fix_y,free_x,free_y))
# solve the system of equations
u = scipy.sparse.linalg.spsolve(G,d,permc_spec='MMD_ATA')
# reshape the solution
u = np.reshape(u,(2,-1))
u_x,u_y = u
## Calculate strain from displacements
D_x = weight_matrix(nodes,nodes,(1,0),n=n)
D_y = weight_matrix(nodes,nodes,(0,1),n=n)
e_xx = D_x.dot(u_x)
e_yy = D_y.dot(u_y)
e_xy = 0.5*(D_y.dot(u_x) + D_x.dot(u_y))
# calculate second strain invariant
I2 = np.sqrt(e_xx**2 + e_yy**2 + 2*e_xy**2)

## Plot the results
#####################################################################
# toss out ghost nodes
g = len(free_idx)
nodes = nodes[:N]
u_x,u_y = u_x[:N],u_y[:N]
I2 = I2[:N]

fig,ax = plt.subplots(figsize=(7,3.5))
# plot the fixed boundary
ax.plot(vert[smp[0],0],vert[smp[0],1],'r-',lw=2,label='fixed',zorder=1)
# plot the free boundary
ax.plot(vert[smp[1],0],vert[smp[1],1],'r--',lw=2,label='free',zorder=1)
for s in smp[2:]:
  ax.plot(vert[s,0],vert[s,1],'r--',lw=2,zorder=1)

# plot the second strain invariant
p = ax.tripcolor(nodes[:,0],nodes[:,1],I2,
                 norm=LogNorm(vmin=0.1,vmax=3.2),
                 cmap='viridis',zorder=0)
# plot the displacement vectors
ax.quiver(nodes[:,0],nodes[:,1],u_x,u_y,zorder=2)
ax.set_xlim((-0.1,2.1))
ax.set_ylim((-0.25,1.1))
ax.set_aspect('equal')
ax.legend(loc=3,frameon=False,fontsize=12,ncol=2)
cbar = fig.colorbar(p)
cbar.set_label('second strain invariant')
fig.tight_layout()
plt.savefig('../figures/fd.b.png')
plt.show() 
