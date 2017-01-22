''' 
In this example we solve the Poisson equation with fix boundary 
conditions on an irregular domain.
'''
import numpy as np
from rbf.basis import phs3
from rbf.domain import logo
from rbf.geometry import contains
from rbf.nodes import menodes
import matplotlib.pyplot as plt

# Define the problem domain. This is done by specifying the vertices of the
# domain, *vert*, and the vertex indices making up each segment, *smp*.
vert,smp = logo()
N = 500 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
edge_idx, = ((smpid>=0) & (smpid<12)).nonzero() # identify edge nodes
eye1_idx, = ((smpid>=12) & (smpid<44)).nonzero() # identify top eye nodes
eye2_idx, = (smpid>=44).nonzero() # identify bottom eye nodes
interior_idx, = (smpid==-1).nonzero() # identify interior nodes
# create left hand side" matrix
A = np.empty((N,N))
A[interior_idx]  = phs3(nodes[interior_idx],nodes,diff=[2,0])
A[interior_idx] += phs3(nodes[interior_idx],nodes,diff=[0,2])
A[edge_idx] = phs3(nodes[edge_idx],nodes)
A[eye1_idx] = phs3(nodes[eye1_idx],nodes)
A[eye2_idx] = phs3(nodes[eye2_idx],nodes)
# set "right hand side" boundary conditions
d = np.zeros(N)
d[eye1_idx] = -1.0
d[eye2_idx] = 1.0
# Solve the PDE
coeff = np.linalg.solve(A,d) # solve for the RBF coefficients
# interpolate the solution on a grid
xg,yg = np.meshgrid(np.linspace(-0.6,1.6,500),np.linspace(-0.6,1.6,500))
points = np.array([xg.flatten(),yg.flatten()]).T                    
u = phs3(points,nodes).dot(coeff) # evaluate at the interp points
u[~contains(points,vert,smp)] = np.nan # mask outside points
ug = u.reshape((500,500)) # fold back into a grid
# make a contour plot of the solution
fig,ax = plt.subplots()
p = ax.contourf(xg,yg,ug,cmap='viridis',vmin=-0.5,vmax=0.5)
ax.plot(nodes[:,0],nodes[:,1],'ko',markersize=4)
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

ax.set_aspect('equal')
ax.set_xlim((-0.6,1.6))
ax.set_ylim((-0.6,1.6))
fig.colorbar(p,ax=ax)
fig.tight_layout()
plt.savefig('../figures/basis.a.png')
plt.show()

