#!/usr/bin/env python
import numpy as np
import rbf.fd
import matplotlib.pyplot as plt
import rbf.basis
import rbf.halton

N = 100
Ns = 10
order = 2
basis = rbf.basis.phs3

x = np.linspace(-2.0,2.0,N)[:,None]
x = np.random.uniform(-2.0,2.0,(N,1))
idx = np.argsort(x[:,0])
x = x[idx]
u = np.sin(x)
udiff_true = -np.sin(x)

L = rbf.fd.diff_matrix(x,(2,),N=Ns,order=order,basis=basis)
udiff = L.dot(u)

fig,ax = plt.subplots()
ax.plot(x[:,0],u[:,0],'b-')
ax.plot(x[:,0],udiff[:,0],'r-')
ax.plot(x[:,0],udiff_true[:,0],'m--')
ax.set_xlim(-2.0,2.0)
plt.show()




