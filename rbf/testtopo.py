#!/usr/bin/env python
import numpy as np
from spect.halton import Halton
from spectral.topology import Domain
import matplotlib.pyplot as plt
import pyximport;pyximport.install()
import nodes
import modest

def curve(t):
  return np.array([0.7 + 0.2*np.cos(t),0.5 + 0.2*np.sin(t)])

points = np.array([curve(i) for i in np.linspace(0,2*np.pi,5)])
print(np.shape(points))
points2 = np.array([points])


D = Domain(points2,['main'])
H = Halton(2)
N = 10000
seq = H(N)
modest.tic()
d = nodes.contains(points,seq)
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



