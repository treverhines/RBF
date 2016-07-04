#!/usr/bin/env python 
import rbf.stencil
import rbf.halton
import matplotlib.pyplot as plt
import numpy as np

N = 50
Ns = 5
nodes = rbf.halton.halton(N,2)

# boundary vertices which a stencil cannot cross
vert = np.array([[0.3,0.2],
                 [0.5,0.4],
                 [0.8,0.5]])
smp = np.array([[0,1],[1,2]])

# stencil network
sn = rbf.stencil.stencil_network(nodes,vert=vert,smp=smp,N=Ns)
# edge network
en = rbf.stencil.stencils_to_edges(sn)

fig,ax = plt.subplots()
ax.set_title('size %s stencils with boundary' % Ns)

# plot boundary
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

# plot edges
for e in en:
  ax.plot(nodes[e,0],nodes[e,1],'k:',lw=1)

# plot nodes
ax.plot(nodes[:,0],nodes[:,1],'ko')


# stencil network
sn = rbf.stencil.stencil_network(nodes,N=Ns)
# edge network
en = rbf.stencil.stencils_to_edges(sn)

fig,ax = plt.subplots()
ax.set_title('size %s stencils without boundary' % Ns)

# plot edges
for e in en:
  ax.plot(nodes[e,0],nodes[e,1],'k:',lw=1)

# plot nodes
ax.plot(nodes[:,0],nodes[:,1],'ko')

plt.show()

