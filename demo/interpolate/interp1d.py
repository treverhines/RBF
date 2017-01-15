#!/usr/bin/env python
# This script demonstrates how to use RBFs for interpolation
import numpy as np
import rbf.basis
import matplotlib.pyplot as plt

### METHOD 1, manually build interpolant ###

# import a prebuilt RBF function
from rbf.basis import phs3

# create 5 observation points
x = np.linspace(-np.pi,np.pi,5)[:,None] 

# find the function value at the observation points
u = np.sin(x[:,0]) 

# create interpolation points
xitp = np.linspace(-4.0,4.0,1000)[:,None] 

# create the coefficient matrix, where each observation point point is 
# an RBF center and each RBF is evaluated at the observation points
A = phs3(x,x)

# find the coefficients for each RBF
coeff = np.linalg.solve(A,u) 

# create the interpolation matrix which evaluates each of the 5 RBFs 
# at the interpolation points
Aitp = rbf.basis.phs3(xitp,x) 

# evaluate the interpolant at the interpolation points
uitp1 = Aitp.dot(coeff)

### METHOD 2, use RBFInterpolant ###
from rbf.interpolate import RBFInterpolant

# This command will produce an identical interpolant to the one 
# created above 
# I = RBFInterpolant(x,u,basis=phs3,order=-1)

# The default values for basis and order are phs3 and 0, where the 
# latter means that a constant term is added to the interpolation 
# function. This tends to produce much better results
I = RBFInterpolant(x,u)
uitp2 = I(xitp)

# plot the results
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x[:,0],u,'ko')
ax.plot(xitp[:,0],uitp1,'r-')
ax.plot(xitp[:,0],uitp2,'b-')
ax.plot(xitp[:,0],np.sin(xitp[:,0]),'k--')
ax.legend(['observation points','interpolant, order=-1','interpolant, order=1','true solution'],loc=2,frameon=False)
fig.tight_layout()
plt.savefig('figures/interp1d.png')
plt.show()
