''' 
this script demonstrates how to use *rbf.stencil.stencil_network*. We 
enforce that no stencil crosses a user defined boundary
'''
import numpy as np
from rbf.stencil import stencil_network
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
colors = ['r','g','b','m','c','y']
np.random.seed(3)

# define the boundary which no stencil can cross
bnd_vert = np.array([[0.1,0.7],[0.7,0.3]])
bnd_smp = np.array([[0,1]])

# create uniformly distributed nodes
nodes = np.random.random((100,2))

# find the 10 nearest neighbors for each node which do not cross the 
# boundary
sn = stencil_network(nodes,nodes,10,vert=bnd_vert,smp=bnd_smp)

# plot the first four stencils
fig,ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(nodes[:,0],nodes[:,1],'ko',ms=5,zorder=0)
for i,s in enumerate(sn[:4]):
  for j in s:
    ax.plot(nodes[[i,j],0],nodes[[i,j],1],'o-',
            c=colors[i%6],ms=6,mec=colors[i%6],zorder=2)
    
  ax.plot(nodes[i,0],nodes[i,1],'o',
          ms=10,c=colors[i%6],mec=colors[i%6],zorder=2)
  hull = ConvexHull(nodes[s])  
  patch = Polygon(nodes[s][hull.vertices],color=colors[i%6],
                  alpha=0.1,edgecolor='none',zorder=1)
  ax.add_patch(patch)

# plot the boundary
for s in bnd_smp:
  plt.plot(bnd_vert[s,0],bnd_vert[s,1],'k-',lw=2)

plt.tight_layout()
plt.savefig('../figures/stencil.a.png')
plt.show()
