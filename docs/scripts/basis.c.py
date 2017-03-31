''' 
In this script we define and plot a Matern RBF.
'''
import numpy as np
from sympy import besselk
from scipy.special import gamma
import matplotlib.pyplot as plt
from rbf.basis import RBF,get_r,get_eps

def make_matern(nu):
  '''returns an instance of a Matern RBF with order *nu*'''
  r,eps = get_r(),get_eps() # get symbolic variables
  coeff = 2**(1 - nu)/gamma(nu) * (np.sqrt(2*nu)*r/eps)**nu
  expr  = coeff*besselk(nu,np.sqrt(2*nu)*r/eps)
  return RBF(expr)

x = np.linspace(0.0,3.0,500)[:,None]
center = np.array([[0.0]])

fig,ax = plt.subplots()
for nu in [0.5,1.5,2.5]:
  mat = make_matern(nu)
  ax.plot(x,mat(x,center),label='$\\nu = $%s' % str(nu))
  
ax.set_xlim((0.0,3.0))
ax.set_ylim((0.0,1.0))
ax.legend()
ax.grid(c='0.5',alpha=0.5)
ax.set_xlabel('$\mathregular{||x - c||_2}$')
plt.tight_layout()
plt.savefig('../figures/basis.c.png')
plt.show()


