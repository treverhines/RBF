''' 
In this script we define and plot an RBF which is based on the sinc 
function
'''
import numpy as np
import sympy
import matplotlib.pyplot as plt
from rbf.basis import RBF,get_R,get_EPS
from mpl_toolkits.mplot3d import Axes3D

R,EPS = get_R(),get_EPS() 
expr = sympy.sin(EPS*R)/(EPS*R) # symbolic expression for the RBF
sinc_rbf = RBF(expr)

# evaluation points
x,y = np.linspace(-5,5,500),np.linspace(-5,5,500)
xg,yg = np.meshgrid(x,y)
xf,yf = xg.ravel(),yg.ravel()
points = np.array([xf,yf]).T

# RBF centers
centers = np.array([[0.0,-3.0],[3.0,2.0],[-2.0,1.0]])

# shape parameters, one for each center
eps = np.array([5.0,5.0,5.0])

# Evaluate the RBFs
values = sinc_rbf(points,centers,eps=eps)

# plot the sum of each RBF
fig,ax = plt.subplots()
p = ax.tripcolor(xf,yf,np.sum(values,axis=1),shading='gouraud',cmap='viridis')
plt.colorbar(p,ax=ax)
ax.set_xlim((-5,5))
ax.set_ylim((-5,5))
plt.tight_layout()
plt.savefig('../figures/basis.b.png')
plt.show()


