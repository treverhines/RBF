#!/usr/bin/env python
# This script demonstrates how the various free parameters for the 
# RBF-FD method affect the quality of the derivative approximation
import numpy as np
import sympy
import rbf.basis
import rbf.fd
import rbf.nodes
import matplotlib.pyplot as plt
import logging
import rbf.domain


logging.basicConfig(level=logging.DEBUG)

Nt = 5000

# RBF-FD FREE PARAMETERS
Ns = None
basis = rbf.basis.phs3
order = None

# define the domain
t = np.linspace(0.0,2*np.pi,100)
x = np.cos(t)
y = np.sin(t)
vert = np.array([x,y]).T
smp = np.array([np.arange(100),np.roll(np.arange(100),-1)]).T
nodes,sid = rbf.nodes.make_nodes(Nt,vert,smp,itr=100)

# define the function
x,y = sympy.symbols('x,y')
r = sympy.sqrt((x-0.0)**2 + y**2)
#u = sympy.exp(-r**2)
#u = 1.0/(1 + r**2)
u = sympy.sin(5*x)*sympy.cos(5*y)
udx = u.diff(x,x)
ufunc = sympy.lambdify((x,y),u,'numpy')
udxfunc = sympy.lambdify((x,y),udx,'numpy')
val = ufunc(nodes[:,0],nodes[:,1])
valdx = udxfunc(nodes[:,0],nodes[:,1])

fig1,ax1 = plt.subplots()
ax1.set_aspect('equal')  
p = ax1.tripcolor(nodes[:,0],nodes[:,1],val)
fig1.colorbar(p)
fig2,ax2 = plt.subplots()
ax2.set_aspect('equal')  
p = ax2.tripcolor(nodes[:,0],nodes[:,1],valdx)
fig2.colorbar(p)

D = rbf.fd.diff_matrix(nodes,(2,0),N=Ns,basis=basis,order=order)
valdx_est = D.dot(val)
fig3,ax3 = plt.subplots()
ax3.set_aspect('equal')  
max_err = np.log10(np.abs(valdx-valdx_est))
print(np.max(max_err[sid==-1]))
#p = ax3.scatter(nodes[:,0],nodes[:,1],c=np.log10(np.abs(valdx-valdx_est)),edgecolor='none')
p = ax3.tripcolor(nodes[:,0],nodes[:,1],valdx_est)
fig3.colorbar(p)
plt.show()  
quit()
