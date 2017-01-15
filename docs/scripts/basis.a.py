''' 
In this example we solve the Poisson equation with a constant forcing 
term using the spectral RBF method.
'''
import numpy as np
from rbf.basis import phs3
from rbf.domain import circle
from rbf.nodes import menodes
import matplotlib.pyplot as plt

# define the problem domain
vert = np.array([[0.762,0.057],[0.492,0.247],[0.225,0.06 ],[0.206,0.056],
                 [0.204,0.075],[0.292,0.398],[0.043,0.609],[0.036,0.624],
                 [0.052,0.629],[0.373,0.63 ],[0.479,0.953],[0.49 ,0.966],
                 [0.503,0.952],[0.611,0.629],[0.934,0.628],[0.95 ,0.622],
                 [0.941,0.607],[0.692,0.397],[0.781,0.072],[0.779,0.055]])

smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],
                [9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
                [16,17],[17,18],[18,19],[19,0]])

N = 500 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
boundary, = (smpid>=0).nonzero() # identify boundary nodes
interior, = (smpid==-1).nonzero() # identify interior nodes

# create left-hand-side matrix and right-hand-side vector
A = np.empty((N,N))
A[interior]  = phs3(nodes[interior],nodes,diff=[2,0])
A[interior] += phs3(nodes[interior],nodes,diff=[0,2])
A[boundary,:] = phs3(nodes[boundary],nodes)
d = np.empty(N)
d[interior] = -100.0
d[boundary] = 0.0

# Solve the PDE
coeff = np.linalg.solve(A,d) # solve for the RBF coefficients
itp = menodes(10000,vert,smp)[0] # interpolation points
soln = phs3(itp,nodes).dot(coeff) # evaluate at the interp points

fig,ax = plt.subplots()
p = ax.scatter(itp[:,0],itp[:,1],s=20,c=soln,edgecolor='none',cmap='viridis')
ax.set_aspect('equal')
ax.plot(nodes[:,0],nodes[:,1],'ko',markersize=4)
ax.set_xlim((0.025,0.975))
ax.set_ylim((0.03,0.98))
plt.colorbar(p,ax=ax)
plt.tight_layout()
plt.savefig('../figures/basis.a.png')
plt.show()

