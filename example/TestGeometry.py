#!/usr/bin/env python
import rbf.geometry
import numpy as np
import mayavi.mlab
import matplotlib.pyplot as plt

vert = np.array([[0.0,0.0],
                 [1.0,0.0],
                 [1.0,1.0],
                 [0.0,1.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,0]])

pnt1 = np.array([[0.5,0.5]])
pnt2 = np.array([[0.5,-1.0]])

#for s in smp:
#  plt.plot(vert[s,0],vert[s,1],'o-')

#plt.plot([pnt1[0,0],pnt2[0,0]],[pnt1[0,1],pnt2[0,1]],'o-')


print(rbf.geometry.contains(pnt1,vert,smp))
print(rbf.geometry.cross_count(pnt1,pnt2,vert,smp))
print(rbf.geometry.intersection_normal(pnt1,pnt2,vert,smp))
print(rbf.geometry.intersection_point(pnt1,pnt2,vert,smp))

#plt.show()

vert = np.array([[0.0,0.0,0.0],
                 [1.0,0.0,0.0],
                 [0.0,1.0,0.0]])
smp = np.array([[0,1,2]])
pnt = np.array([[0.5,0.1,-1.0],
                [0.2,0.2,0.0]])
mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp)
mayavi.mlab.plot3d(pnt[:,0],pnt[:,1],pnt[:,2])
print(rbf.geometry.intersection_point(pnt[[0]],pnt[[1]],vert,smp))
print(rbf.geometry.intersection_normal(pnt[[0]],pnt[[1]],vert,smp))
print(rbf.geometry.intersection_index(pnt[[0]],pnt[[1]],vert,smp))

mayavi.mlab.show()



