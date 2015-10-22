#!/usr/bin/env python
import numpy as np
from rbf.halton import Halton
import matplotlib.pyplot as plt
import rbf.spatial as spatial
import modest

def curve(t):
  return np.array([0.7 + 0.2*np.cos(t),0.5 + 0.2*np.sin(t)])

points = np.array([curve(i) for i in np.linspace(0,2*np.pi,1000)])
print(np.shape(points))
points2 = np.array([points])


H = Halton(2)
N = 1000000
modest.tic()
seq = H(N)
print(modest.toc())

modest.tic()
d = spatial.contains(points,seq)
print(modest.toc())

#modest.tic()
#d = D.contains(seq)
#print(modest.toc())

#print(d)
plt.plot(seq[d,0],seq[d,1],'o')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

plt.plot(seq[~d,0],seq[~d,1],'o')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()



