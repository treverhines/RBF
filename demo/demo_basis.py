#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.basis

# demonstrate instantiation and evaluation of RBF
R = rbf.basis.get_R()
expr = 1/(1 + R**2) # inverse quadratic
iq = rbf.basis.RBF(expr)

c = np.array([[-2.0],[0.0],[2.0]]) # RBF centers
x = np.linspace(-5.0,5.0,1000)[:,None] # evaluate at these points

soln = iq(x,c)

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,soln)
ax.set_xlim((-5.0,5.0))
ax.set_ylim((0.0,1.1))
ax.set_title('inverse quadratic radial basis functions')
ax.grid()
fig.tight_layout()
plt.savefig('figures/demo_basis.png')






