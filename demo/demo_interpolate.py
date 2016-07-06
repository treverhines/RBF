#!/usr/bin/env python
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt
from rbf.basis import iq

x = np.linspace(-np.pi,np.pi,5)[:,None] # observation points
u = np.sin(x[:,0]) # values
xitp = np.linspace(-4.0,4.0,1000)[:,None] # interpolation points

A = rbf.basis.phs3(x,x)
coeff = np.linalg.solve(A,u) # estimate coefficients for each RBF
Aitp = rbf.basis.phs3(xitp,x) # interpolation matrix
uitp = Aitp.dot(coeff)

fig,ax = plt.subplots()
ax.plot(x[:,0],u,'ko')
ax.plot(xitp[:,0],uitp,'k-')
ax.plot(xitp[:,0],np.sin(xitp[:,0]),'k--')
ax.legend(['observation points','interpolant','true solution'],loc=2,frameon=False)
ax.set_title('third-order polyharmonic spline interpolation')
fig.tight_layout()
plt.savefig('figures/interpolate_demo.png')
