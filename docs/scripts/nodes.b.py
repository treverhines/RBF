'''
This script demonstrates generating nodes with `min_energy_nodes` and
it verifies that the density of the returned nodes accurately matches
the specified density.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.pde.nodes import min_energy_nodes
import rbf
import logging
logging.basicConfig(level=logging.DEBUG)


def desired_rho(x):
  '''
  desired node density
  '''
  r1 = np.sqrt((x[:, 0] - 0.25)**2 + (x[:, 1] - 0.25)**2)
  bump1 = 0.76 / ((r1/0.1)**2 + 1.0)

  r2 = np.sqrt((x[:, 0] - 0.25)**2 + (x[:, 1] - 0.75)**2)
  bump2 = 0.76 / ((r2/0.1)**4 + 1.0)

  r3 = np.sqrt((x[:, 0] - 0.75)**2 + (x[:, 1] - 0.75)**2)
  bump3 = 0.76 / ((r3/0.1)**8 + 1.0)

  r4 = np.sqrt((x[:, 0] - 0.75)**2 + (x[:, 1] - 0.25)**2)
  bump4 = 0.76 / ((r4/0.1)**16 + 1.0)

  out = 0.2 + bump1 + bump2 + bump3 + bump4
  return out


def actual_rho(x, nodes):
  '''
  compute the density of `nodes` and evaluate the density function at
  `x`. The output is normalize 1.0
  '''
  out = np.zeros(x.shape[0])
  for n in nodes:
    out += rbf.basis.se(x, n[None, :], eps=0.01)[:, 0]
    
  out /= np.max(out)
  return out

vert = np.array([[0.0, 0.0],
                 [1.0, 0.0],
                 [1.0, 1.0],
                 [0.0, 1.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])                 

nodes = min_energy_nodes(10000, vert, smp, rho=desired_rho)[0]

# plot the nodes
fig, ax = plt.subplots()
for s in smp:
  ax.plot(vert[s, 0], vert[s, 1], 'k-')

ax.plot(nodes[:, 0], nodes[:, 1], 'k.', ms=1)  
ax.set_aspect('equal')
ax.set_title('node positions')
fig.tight_layout()
plt.savefig('../figures/nodes.b.1.png')

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# evaluate and plot the node density
x = np.linspace(-0.1, 1.1, 250)
y = np.linspace(-0.1, 1.1, 250)
xg, yg = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()
points = np.array([x, y]).T

p = axs[0].tripcolor(points[:, 0], points[:, 1], desired_rho(points),
                     cmap='viridis', vmin=0.0, vmax=1.0)
for s in smp:
  axs[0].plot(vert[s, 0], vert[s, 1], 'w-')

axs[0].set_xlim(-0.1, 1.1)
axs[0].set_ylim(-0.1, 1.1)
axs[0].set_aspect('equal')
axs[0].set_title('desired node density')
fig.colorbar(p, ax=axs[0])

p = axs[1].tripcolor(points[:, 0], points[:, 1], actual_rho(points, nodes),
                     cmap='viridis', vmin=0.0, vmax=1.0)
for s in smp:
  axs[1].plot(vert[s, 0], vert[s, 1], 'w-')

axs[1].set_xlim(-0.1, 1.1)
axs[1].set_ylim(-0.1, 1.1)
axs[1].set_aspect('equal')
axs[1].set_title('actual node density')
fig.colorbar(p, ax=axs[1])
fig.tight_layout()
plt.savefig('../figures/nodes.b.2.png')
plt.show()


