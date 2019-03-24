'''
This script demonstrates how to generate poisson discs with variable
radii.
'''
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rbf.pde.knn import k_nearest_neighbors
from rbf.pde.sampling import poisson_discs


def radius(x):
    '''disc radius as a function of position'''
    r = np.linalg.norm(x - 0.5, axis=1)
    return 0.02 + 0.2*r

# define the domain
vert = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

# generate the discs
centers = poisson_discs(radius, vert, smp)

# plot the domain and discs
fig, ax = plt.subplots()
for s in smp:
    ax.plot(vert[s, 0], vert[s, 1], 'k-')

for c, r in zip(centers, radius(centers)):
    ax.plot(c[0], c[1], 'C0.')
    ax.add_artist(Circle(c, r, color='C0', alpha=0.1))

ax.set_aspect('equal')
ax.set_title('Poisson discs with variable radii')
plt.tight_layout()
plt.savefig('../figures/sampling.a.1.png')

# plot the nearest neighbor distance divided by the disc radii. This
# verifies that all nodes are spaced sufficiently far away
fig, ax = plt.subplots()
dist = k_nearest_neighbors(centers, centers, 2)[1][:, 1]
ax.hist(dist/radius(centers), 10)
ax.set_xlabel('(nearest neighbor distance) / (disc radius)')
ax.set_ylabel('count')
ax.grid(ls=':', color='k')
plt.tight_layout()
plt.savefig('../figures/sampling.a.2.png')
plt.show()
                    
