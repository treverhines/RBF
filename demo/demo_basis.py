#!/usr/bin/env python
# this script demonstrates how to create and evaluate RBFs
import numpy as np
import matplotlib.pyplot as plt
import rbf.basis

# demonstrate instantiation and evaluation of RBF
R = rbf.basis.get_R()
expr = 1/(1 + R**2) # inverse quadratic
iq = rbf.basis.RBF(expr)

# create RBF centers
# indexing with 'None' changes it from a (N,) array to a (N,1) array
c = np.array([-2.0,0.0,2.0])[:,None] 
# create evaluation points
x = np.linspace(-5.0,5.0,1000)[:,None] 

# evaluate each RBF at x
soln = iq(x,c)
# evaluate the first derivative of each RBF at x
soln_diff = iq(x,c,diff=(1,))

# plot results
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic')
ax.grid()
fig.tight_layout()
plt.savefig('figures/demo_basis_1.png')

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln_diff)
ax.set_xlim((-5.0,5.0))
ax.set_title('inverse quadratic first derivative')
ax.grid()
fig.tight_layout()
plt.savefig('figures/demo_basis_2.png')
plt.show()
