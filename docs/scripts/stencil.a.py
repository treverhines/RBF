''' 
this script demonstrates how to use *rbf.stencil.stencil_network*. We 
enforce that no stencil crosses a user defined boundary
'''
import numpy as np
from rbf.stencil import stencil_network
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
colors = ['r','g','b','m','c','y']
np.random.seed(3)

# define the boundary which no stencil can cross
bnd_vert = np.array([[0.1,0.8],
                     [0.7,0.3]])
bnd_smp = np.array([[0,1]])


nodes = np.random.random((100,2))

sn = stencil_network(nodes,10,vert=bnd_vert,smp=bnd_smp)


fig,ax = plt.subplots()
ax.plot(nodes[:,0],nodes[:,1],'ko',zorder=0)
for i,s in enumerate(sn):
  hull = ConvexHull(nodes[s])  
  patch = Polygon(nodes[s][hull.vertices],color=colors[i%6],
                  alpha=0.2,edgecolor='none',zorder=1)
  ax.add_patch(patch)
  ax.scatter(nodes[i,0],nodes[i,1],s=50,c=colors[i%6],edgecolor='none',zorder=2)

for s in bnd_smp:
  plt.plot(bnd_vert[s,0],bnd_vert[s,1],'k-',lw=2)

plt.show()
