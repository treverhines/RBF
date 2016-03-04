#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.geometry
import mayavi.mlab
import time
try:
  import gts
except ImportError:
  print('cannot import gts. This package is needed for the 3D point '
        'in polyhedra demonstration. It can be installed with "pip '
        'install pygts"')

# 2D
######################################################################
N = 100000
vert = np.array([[0.0,0.0],
                 [0.5,0.2],
                 [1.0,1.0],
                 [0.25,0.5]])
smp = np.array([[0,1],[1,2],[2,3],[3,0]])
point = np.random.random((1,2))

for s in smp:
  plt.plot(vert[s,0],vert[s,1],'ko-')

plt.plot(point[:,0],point[:,1],'bo')
print('is inside: %s' % rbf.geometry.contains(point,vert,smp)[0])
point = (np.random.random((N,2))-0.5)*3
start = time.time()
contains = rbf.geometry.contains(point,vert,smp)
stop = time.time()
plt.plot(point[contains,0],point[contains,1],'bo',markersize=1)
print('time per point tested: %s ns' % ((stop-start)*1e9/N))
plt.show()

# 3D
######################################################################
# the following is just used to get the vertices and simplices
f = open('icosa.gts')
s = gts.read(f)
f.close()
x,y,z,t = gts.get_coords_and_face_indices(s,True)
vert = np.array([x,y,z]).T
smp = np.array(t)
point = (np.random.random((1,3))-0.5)*5
mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,color=(1.0,1.0,1.0),opacity=0.5)
mayavi.mlab.points3d(point[:,0],point[:,1],point[:,2],scale_factor=0.5,color=(1.0,0.0,0.0))
print('is inside: %s' % rbf.geometry.contains(point,vert,smp)[0])
point = (np.random.random((N,3))-0.5)*5
start = time.time()
contains = rbf.geometry.contains(point,vert,smp)
stop = time.time()
mayavi.mlab.points3d(point[contains,0],point[contains,1],point[contains,2],scale_factor=0.1,color=(1.0,0.0,0.0))

print('time per point tested: %s ns' % ((stop-start)*1e9/N))
mayavi.mlab.show()







