''' 
In this script we solve the 2-d wave equation with a RBF-FD scheme
'''
import numpy as np
from rbf.fd import weight_matrix
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
dt = 0.000025 # time step size
N = 100000 # total number of nodes
nodes,smpid = menodes(N,vert,smp) # generate nodes
boundary, = (smpid>=0).nonzero() # identify boundary nodes
interior, = (smpid==-1).nonzero() # identify interior nodes
D = weight_matrix(nodes[interior],nodes,[[2,0],[0,2]],n=30)
r = np.linalg.norm(nodes-np.array([0.49,0.46]),axis=1)
u_prev = 1.0/(1 + (r/0.01)**4) # create initial conditions
u_curr = 1.0/(1 + (r/0.01)**4)
fig,axs = plt.subplots(2,2,figsize=(7,7))
axs = [axs[0][0],axs[0][1],axs[1][0],axs[1][1]]
for i in range(15001):
  u_next = dt**2*D.dot(u_curr) + 2*u_curr[interior] - u_prev[interior]
  u_prev[:] = u_curr
  u_curr[interior] = u_next
  if i in [0,5000,10000,15000]:  
    ax = axs[[0,5000,10000,15000].index(i)]
    p = ax.scatter(nodes[:,0],nodes[:,1],s=3,c=np.array(u_curr,copy=True),
                   edgecolor='none',cmap='viridis',vmin=-0.1,vmax=0.1)
    for s in smp: ax.plot(vert[s,0],vert[s,1],'k-')
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False);ax.get_yaxis().set_visible(False)
    ax.set_xlim((0.025,0.975));ax.set_ylim((0.03,0.98))
    ax.text(0.57,0.85,'time : %s\nnodes : %s' % (np.round(i*dt,1),N),
            transform=ax.transAxes,fontsize=12)

plt.tight_layout()    
#plt.savefig('../figures/fd.a.png')
plt.show()

