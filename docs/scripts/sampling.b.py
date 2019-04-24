import logging
import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from rbf.pde.knn import k_nearest_neighbors
from rbf.pde.sampling import rejection_sampling
from rbf.basis import se

logging.basicConfig(level=logging.DEBUG)

def rho(x):
    r = np.linalg.norm(x - 0.5, axis=1)
    return 1.0/(1.0 + (r/0.25)**2)

# define the domain
vert = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

# generate the discs
start = time.time()
points = rejection_sampling(1000, rho, (vert, smp))
print('generated %s nodes in %s seconds' %
      (len(points), time.time() - start))

# plot the domain and discs
fig, ax = plt.subplots()
for s in smp:
    ax.plot(vert[s, 0], vert[s, 1], 'k-')

ax.plot(points[:, 0], points[:, 1], 'ko')

ax.set_aspect('equal')
ax.set_title('Rejection sampling points')
plt.tight_layout()
plt.show()

