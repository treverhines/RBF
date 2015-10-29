#!/usr/bin/env python
import numpy as np
from rbf.halton import Halton
import matplotlib.pyplot as plt
import rbf.spatial as spatial
import modest
import scipy.spatial

def curve(t):
  return np.array([0.7 + 0.2*np.cos(t),0.5 + 0.2*np.sin(t)])

e1 = np.array([[0.,0.],[2.0,2.0]])
e2 = np.array([[3.0,3.5],[1.9,2.1]])
plt.plot(e1[:,0],e1[:,1],'b-o')
plt.plot(e2[:,0],e2[:,1],'r-o')
plt.xlim(-5,5)
plt.ylim(-5,5)
print('intersecting %s' % str(spatial.is_intersecting(e1,e2)))
print('collinear %s' % spatial.is_collinear(e1,e2))
print('overlapping %s' % spatial.is_overlapping(e1,e2))
print('parallel %s' % spatial.is_parallel(e1,e2))
plt.show()

points = np.array([curve(i) for i in np.linspace(0,1.99*np.pi,100)])
#print(np.shape(points))
#points = np.array([[0.5,0.5],
#                   [0.7,0.5]])
#points2 = np.array([points])


H = Halton(2)
N = 100000
modest.tic()
seq = H(N)
print(modest.toc())

modest.tic()
d = spatial.contains(seq[:,[0,1]],points)
print(d)
print(modest.toc())
modest.tic()
T = scipy.spatial.cKDTree(seq)
T.query(seq,5)
print(modest.toc())

plt.plot(seq[d,0],seq[d,1],'o')
plt.show()
plt.plot(seq[~d,0],seq[~d,1],'o')
plt.show()
#modest.tic()
#d = D.contains(seq)
#print(modest.toc())

#print(d)




