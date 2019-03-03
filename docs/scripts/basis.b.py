''' 
In this script we define and plot an RBF which is based on the sinc 
function
'''
import numpy as np
import sympy
import matplotlib.pyplot as plt
from rbf.basis import RBF,get_r,get_eps

r, eps = get_r(), get_eps() # get symbolic variables
expr = sympy.sin(eps*r)/(eps*r) # symbolic expression for the RBF
sinc_rbf = RBF(expr) # instantiate the RBF
x = np.linspace(-5, 5, 500)
points = np.reshape(np.meshgrid(x, x), (2, 500*500)).T # interp points
centers = np.array([[0.0, -3.0], [3.0, 2.0], [-2.0, 1.0]]) # RBF centers
eps = np.array([5.0, 5.0, 5.0]) # shape parameters
values = sinc_rbf(points, centers, eps=eps) # Evaluate the RBFs

# plot the sum of each RBF
fig, ax = plt.subplots()
p = ax.tripcolor(points[:, 0], points[:, 1], np.sum(values, axis=1),
                 shading='gouraud', cmap='viridis')
plt.colorbar(p, ax=ax)
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
plt.tight_layout()
plt.savefig('../figures/basis.b.png')
plt.show()


