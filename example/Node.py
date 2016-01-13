#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.nodegen
import rbf.normalize
import logging
logging.basicConfig(level=logging.INFO)

# 1D Node Generation
vert = np.array([[-1.0],[2.0]])
smp = np.array([[0],[1]])

N = 200
@rbf.normalize.normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  return 1.0/(1.0 + 10*p[:,0]**2)

nodes,norms,groups = rbf.nodegen.volume(rho,vert,smp)
plt.figure(1)
plt.plot(nodes[:,0],0*nodes[:,0],'o')

plt.figure(2)
plt.hist(nodes[:,0],20,normed=True)
x = np.linspace(-1.0,2.0,1000)
plt.plot(x,rho(x[:,None])/N)


plt.figure(3)
x,dx = rbf.stencil.nearest(nodes,nodes,2)
plt.hist(dx[:,1]*rho(nodes),20)

plt.show()

