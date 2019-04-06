'''
This script demonstrates when it is more efficient to use the Quad/Oct
trees when detection boundary collisions with `intersection_count`
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.pde.domain import sphere
from rbf.pde.geometry import (intersection_count, 
                              intersection_count_rtree)
import time

segments = 10000
simplices = []
time_with_tree = []
time_tree_build = []
time_without_tree = []

# create a collection of small line segments. 
point1 = np.random.uniform(-1.0, 1.0, (segments, 3))
point2 = point1 + np.random.normal(0.0, 1.0, (segments, 3))
for r in range(7):
    vert, smp = sphere(r)
    simplices += [len(smp)]
    
    start = time.time()
    intersection_count(point1, point2, vert, smp)
    stop = time.time()
    time_without_tree += [stop - start]
    
    start2 = time.time()
    start1 = time.time()
    tree = intersection_count_rtree(vert, smp)
    stop1 = time.time()
    time_tree_build += [stop1 - start1]

    intersection_count(point1, point2, vert, smp, tree=tree)
    stop2 = time.time()
    time_with_tree += [stop2 - start2]
    
print(time_with_tree)
fig, ax = plt.subplots()
ax.loglog(simplices, time_with_tree, 'C0o-', label='with tree')
ax.loglog(simplices, time_tree_build, 'C1o-', label='tree build')
ax.loglog(simplices, time_without_tree, 'C2o-', label='without tree')
ax.grid(ls=':')
ax.set_ylabel('time [seconds]')
ax.set_xlabel('simplices')
ax.legend()
plt.show()
