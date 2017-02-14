''' 
In this script we solve the 2-d wave equation over an L-shaped domain 
with a RBF-FD scheme.
'''
import numpy as np
from rbf.fd import weight_matrix
from rbf.nodes import menodes
from rbf.geometry import contains
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# define the problem domain
vert = np.array([[0.0,0.0],[2.0,0.0],[2.0,1.0],
                 [1.0,1.0],[1.0,2.0],[0.0,2.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
dt = 0.001 # time step size
N = 1000 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
boundary, = (smpid>=0).nonzero() # identify boundary nodes
interior, = (smpid==-1).nonzero() # identify interior nodes
D = weight_matrix(nodes[interior],nodes,[[2,0],[0,2]],n=500)
Deig = weight_matrix(nodes,nodes,[[2,0],[0,2]],n=500)
val,vec = np.linalg.eig(Deig.toarray())
fig,ax = plt.subplots()
ax.grid(True)
ax.plot(val.real,val.imag,'k.')
plt.show()
quit()
r = np.linalg.norm(nodes-np.array([0.5,0.5]),axis=1)
u_prev = 0.5/(1 + (r/0.2)**4) # create initial conditions
u_curr = 0.5/(1 + (r/0.2)**4)
fig,axs = plt.subplots(2,2,figsize=(7,7))
axs = [axs[0][0],axs[0][1],axs[1][0],axs[1][1]]
plot_steps = [200,600,1000,1400]
for i in range(1401):
  u_next = dt**2*D.dot(u_curr) + 2*u_curr[interior] - u_prev[interior]
  u_prev[interior] = u_curr[interior]
  u_curr[interior] = u_next
  if i in plot_steps:  
    # plot the solution at specified time steps
    ax = axs[plot_steps.index(i)]
    xg,yg = np.mgrid[0.0:2.0:400j,0:2.0:400j]
    points = np.array([xg.ravel(),yg.ravel()]).T
    ug = griddata(nodes,u_curr,(xg,yg),method='linear')
    ug.ravel()[~contains(points,vert,smp)] = np.nan 
    for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
    ax.contourf(xg,yg,ug,cmap='viridis',vmin=-0.2,vmax=0.2,
                levels=np.linspace(-0.25,0.25,14))
    ax.set_aspect('equal')
    ax.text(0.6,0.85,'time : %s\nnodes : %s' % (np.round(i*dt,3),N),
            transform=ax.transAxes,fontsize=10)
    ax.tick_params(labelsize=10)
    ax.set_xlim(-0.1,2.1)
    ax.set_ylim(-0.1,2.1)
    
plt.tight_layout()    
plt.savefig('../figures/fd.a.png')
plt.show()

