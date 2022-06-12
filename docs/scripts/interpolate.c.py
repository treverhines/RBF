'''
This script compares RBFInterpolant to NearestRBFInterpolant class for Franke's
test function.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.interpolate import RBFInterpolant, KNearestRBFInterpolant
np.random.seed(1)

def frankes_test_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y

# the observations and their locations for interpolation
xobs = np.random.uniform(0.0, 1.0, (500, 2))
yobs = frankes_test_function(xobs)

# the locations where we evaluate the interpolants
xitp = np.mgrid[0:1:200j, 0:1:200j].reshape(2, -1).T

# the true function which we want the interpolants to reproduce
true_soln = frankes_test_function(xitp)

yitp = RBFInterpolant(xobs, yobs)(xitp)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].set_title('RBFInterpolant')
p = ax[0].tripcolor(xitp[:, 0], xitp[:, 1], yitp)
ax[0].scatter(xobs[:, 0], xobs[:, 1], c='k', s=3)
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)
ax[0].set_aspect('equal')
ax[0].grid(ls=':', color='k')
fig.colorbar(p, ax=ax[0])
ax[1].set_title('|error|')
p = ax[1].tripcolor(xitp[:, 0], xitp[:, 1], np.abs(yitp - true_soln))
ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, 1)
ax[1].set_aspect('equal')
ax[1].grid(ls=':', color='k')
fig.colorbar(p, ax=ax[1])
fig.tight_layout()
plt.savefig('../figures/interpolate.c.all.png')

for k in [5, 20, 50]:
    yitp = RBFInterpolant(xobs, yobs, neighbors=k)(xitp)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].set_title('RBFInterpolant with neighbors=%d' % k)
    p = ax[0].tripcolor(xitp[:, 0], xitp[:, 1], yitp)
    ax[0].scatter(xobs[:, 0], xobs[:, 1], c='k', s=3)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_aspect('equal')
    ax[0].grid(ls=':', color='k')
    fig.colorbar(p, ax=ax[0])
    ax[1].set_title('|error|')
    p = ax[1].tripcolor(xitp[:, 0], xitp[:, 1], np.abs(yitp - true_soln))
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_aspect('equal')
    ax[1].grid(ls=':', color='k')
    fig.colorbar(p, ax=ax[1])
    fig.tight_layout()
    plt.savefig('../figures/interpolate.c.%d.png' % k)

plt.show()
