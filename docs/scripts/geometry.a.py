'''
This script demonstrates when it is more efficient to use the Quad/Oct
trees when detection boundary collisions with `intersection_count`
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.pde.domain import sphere
from rbf.pde.geometry import intersection_count
import time

segments = 1000
simplices = []
time_with_tree = []
time_without_tree = []

# create a collection of small line segments. 
point1 = np.random.uniform(-1.0, 1.0, (segments, 3))
point2 = point1 #+ np.random.normal(0.0, 0.001, (segments, 3))
for r in range(9):
    vert, smp = sphere(r)
    simplices += [len(smp)]
    
    start = time.time()
    intersection_count(point1, point2, vert, smp, use_qotree=True)
    stop = time.time()
    time_with_tree += [stop - start]
    
    start = time.time()
    intersection_count(point1, point2, vert, smp, use_qotree=False)
    stop = time.time()
    time_without_tree += [stop - start]
    
print(time_with_tree)
fig, ax = plt.subplots()
ax.loglog(simplices, time_with_tree, 'C0o-', label='with oct-tree')
ax.loglog(simplices, time_without_tree, 'C1o-', label='without oct-tree')
ax.grid(ls=':')
ax.set_ylabel('time [seconds]')
ax.set_xlabel('simplices')
ax.legend()
plt.show()
