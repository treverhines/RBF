#!/usr/bin/env python
# Interpolate the function sin(5*x) + cos(8*y) at scattered points
import numpy as np
import matplotlib.pyplot as plt
import rbf.halton
import rbf.basis
from rbf.interpolate import RBFInterpolant
import matplotlib.cm

def f(x):
  return np.sin(5*x[:,0]) + np.cos(8*x[:,1])

N = 20
Nitp = 10000
x = rbf.halton.halton(N,2)
xitp = rbf.halton.halton(Nitp,2)

val = f(x)
#val += np.random.normal(0.0,0.5,val.shape)

valitp = f(xitp)

A = RBFInterpolant(x,val,order=1,basis=rbf.basis.phs3,extrapolate=True,penalty=0.01)
pred = A(xitp)

# true solution
plt.figure(1)
plt.tripcolor(xitp[:,0],xitp[:,1],valitp,cmap=matplotlib.cm.cubehelix,vmin=-2,vmax=2)

plt.figure(2)
plt.tripcolor(xitp[:,0],xitp[:,1],pred,cmap=matplotlib.cm.cubehelix,vmin=-2,vmax=2)
plt.scatter(x[:,0],x[:,1],s=80,c=val,cmap=matplotlib.cm.cubehelix,vmin=-2,vmax=2)
plt.show()
