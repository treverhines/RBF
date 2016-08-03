#!/usr/bin/env python
from PIL import Image
import numpy as np
from rbf.integrate import density_normalizer
import rbf.nodegen
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

img = Image.open('Lenna.png')
imga = np.array(img,dtype=float)/256.0
c = np.linalg.norm(imga,axis=-1)

N = 50000

t = np.linspace(0,2*np.pi,201)
t = t[:-1]
radius = 0.45*(0.1*np.sin(10*t) + 1.0)
vert = np.array([0.5+radius*np.cos(t),
                 0.5+radius*np.sin(t)]).T
smp = np.array([np.arange(200),np.roll(np.arange(200),-1)]).T

@density_normalizer(vert,smp,N)
def rho(p):
  p = p*512
  p = np.array(p,dtype=int)
  return np.max(c)+1e-6 - c[511-p[:,1],p[:,0]]

nodes,smpid = rbf.nodegen.volume(rho,vert,smp)
is_boundary = smpid >= 0
is_interior = smpid == -1

plt.plot(nodes[is_interior,0],nodes[is_interior,1],'k.',markersize=3)
plt.plot(nodes[is_boundary,0],nodes[is_boundary,1],'b.',markersize=5)
plt.show()

