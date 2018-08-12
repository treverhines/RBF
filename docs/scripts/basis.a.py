''' 
In this example we solve the Poisson equation over an L-shaped domain 
with fixed boundary conditions. We use the multiquadratic RBF (`mq`) 
'''
import numpy as np
from rbf.basis import mq
from rbf.geometry import contains
from rbf.nodes import min_energy_nodes
import matplotlib.pyplot as plt

# Define the problem domain with line segments.
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
N = 500 # total number of nodes
nodes,idx,_ = min_energy_nodes(N,vert,smp) # generate nodes
eps = 5.0  # shape parameter
# create "left hand side" matrix
A = np.empty((N,N))
A[idx['interior']]  = mq(nodes[idx['interior']],nodes,eps=eps,diff=[2,0])
A[idx['interior']] += mq(nodes[idx['interior']],nodes,eps=eps,diff=[0,2])
A[idx['boundary']]  = mq(nodes[idx['boundary']],nodes,eps=eps)
# create "right hand side" vector
d = np.empty(N)
d[idx['interior']] = -1.0 # forcing term
d[idx['boundary']] = 0.0 # boundary condition
# Solve for the RBF coefficients
coeff = np.linalg.solve(A,d) 
# interpolate the solution on a grid
xg,yg = np.meshgrid(np.linspace(-0.05,2.05,400),np.linspace(-0.05,2.05,400))
points = np.array([xg.flatten(),yg.flatten()]).T                    
u = mq(points,nodes,eps=eps).dot(coeff) # evaluate at the interp points
u[~contains(points,vert,smp)] = np.nan # mask outside points
ug = u.reshape((400,400)) # fold back into a grid
# make a contour plot of the solution
fig,ax = plt.subplots()
p = ax.contourf(xg,yg,ug,np.linspace(0.0,0.16,9),cmap='viridis')
ax.plot(nodes[:,0],nodes[:,1],'ko',markersize=4)
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

ax.set_aspect('equal')
fig.colorbar(p,ax=ax)
fig.tight_layout()
plt.savefig('../figures/basis.a.png')
plt.show()

