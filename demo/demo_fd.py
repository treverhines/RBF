#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.halton
import rbf.interpolate
import matplotlib.pyplot as plt

# 1D
x = np.linspace(-5,5,10)[:,None]
xitp = np.linspace(-5,5,1000)[:,None]
u = np.sin(x[:,0])
uitp = rbf.fd.weight_matrix(xitp,x,diff=(0,),N=10,order=1).dot(u)

fig,ax = plt.subplots()
ax.plot(x[:,0],u,'ko')
ax.plot(xitp[:,0],uitp,'k-')
ax.plot(xitp[:,0],np.sin(xitp[:,0]),'k--')

# 2D
x = rbf.halton.halton(50,2)
xitp = rbf.halton.halton(10000,2)
u = np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
uitp = rbf.fd.weight_matrix(xitp,x,diff=(0,0),N=10,order=1).dot(u)

fig,ax = plt.subplots()
p = ax.tripcolor(xitp[:,0],xitp[:,1],uitp)
ax.scatter(x[:,0],x[:,1],s=100,c=u,vmin=-1.0,vmax=1.0)
fig.colorbar(p,ax=ax)

plt.show()
