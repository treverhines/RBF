#!/usr/bin/env python
# This script demonstrates how the stencil size affects the accuracy 
# of the RBF-FD approximation
import numpy as np
import sympy
import rbf.basis
import rbf.fd
import rbf.nodes
import matplotlib.pyplot as plt
import logging
import rbf.domain
logging.basicConfig(level=logging.DEBUG)

# create test function
x,y = sympy.symbols('x,y')
f = 1 + sympy.sin(4*x) + sympy.cos(3*x) + sympy.sin(2*y)
Lf = f.diff(x) + f.diff(y)
f = sympy.lambdify((x,y),f,'numpy')
Lf = sympy.lambdify((x,y),Lf,'numpy')

# create nodes
T = 1000
vert,smp = rbf.domain.circle()
nodes,sid = rbf.nodes.make_nodes(T,vert,smp,neighbors=5,itr=200,delta=0.05)
interior, = np.nonzero(sid == -1)
boundary, = np.nonzero(sid >= 0)

# plot nodes 
fig,ax = plt.subplots()
ax.plot(nodes[:,0],nodes[:,1],'ko')
for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-')

ax.set_aspect('equal')
fig.tight_layout()

# plot test function
val = f(nodes[:,0],nodes[:,1])
diff_true = Lf(nodes[:,0],nodes[:,1])
p = ax.tripcolor(nodes[:,0],nodes[:,1],diff_true,cmap='viridis')
ax.set_title(u'$\Delta$ u(x,y)')
fig.colorbar(p)

# Stencil Size = 6
fig,ax = plt.subplots(2,2)
ax[0][0].set_title('stencil size = 3')
ax[0][0].set_aspect('equal')
N = 3
L = rbf.fd.diff_matrix(nodes,[[1,0],[0,1]],size=N)
diff_est = L.dot(val)
err = np.abs(diff_est - diff_true)

p = ax[0][0].tripcolor(nodes[:,0],nodes[:,1],np.log10(err),cmap='viridis')
for s in smp:
  ax[0][0].plot(vert[s,0],vert[s,1],'k-')

cbar = fig.colorbar(p,ax=ax[0][0])
cbar.set_label('log10(error)')

# Stencil Size = 10
ax[0][1].set_title('stencil size = 5')
ax[0][1].set_aspect('equal')
N = 5
L = rbf.fd.diff_matrix(nodes,[[1,0],[0,1]],size=N)
diff_est = L.dot(val)
err = np.abs(diff_est - diff_true)

p = ax[0][1].tripcolor(nodes[:,0],nodes[:,1],np.log10(err),cmap='viridis')
for s in smp:
  ax[0][1].plot(vert[s,0],vert[s,1],'k-')

cbar = fig.colorbar(p,ax=ax[0][1])
cbar.set_label('log10(error)')

# Stencil Size = 20
ax[1][0].set_title('stencil size = 10')
ax[1][0].set_aspect('equal')
N = 10
L = rbf.fd.diff_matrix(nodes,[[1,0],[0,1]],size=N)
diff_est = L.dot(val)
err = np.abs(diff_est - diff_true)

p = ax[1][0].tripcolor(nodes[:,0],nodes[:,1],np.log10(err),cmap='viridis')
for s in smp:
  ax[1][0].plot(vert[s,0],vert[s,1],'k-')

cbar = fig.colorbar(p,ax=ax[1][0])
cbar.set_label('log10(error)')

# Stencil Size = 30
ax[1][1].set_title('stencil size = 15')
ax[1][1].set_aspect('equal')
N = 15
L = rbf.fd.diff_matrix(nodes,[[1,0],[0,1]],size=N)
diff_est = L.dot(val)
err = np.abs(diff_est - diff_true)

p = ax[1][1].tripcolor(nodes[:,0],nodes[:,1],np.log10(err),cmap='viridis')
for s in smp:
  ax[1][1].plot(vert[s,0],vert[s,1],'k-')

cbar = fig.colorbar(p,ax=ax[1,1])
cbar.set_label('log10(error)')

fig.tight_layout()

# compute max error as a function of stencil size
sizes = range(3,60)
max_err = np.zeros(len(sizes))
for i,s in enumerate(sizes):
  L = rbf.fd.diff_matrix(nodes,[[1,0],[0,1]],size=s)
  diff_est = L.dot(val)
  err = np.abs(diff_est - diff_true)
  max_err[i] = np.max(err)

fig,ax = plt.subplots()
ax.set_ylabel('maximum error')
ax.set_xlabel('stencil size')
ax.semilogy(sizes,max_err)  
ax.grid()

fig.tight_layout()
plt.show()
   

