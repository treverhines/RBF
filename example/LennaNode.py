#!/usr/bin/env python
from PIL import Image
import numpy as np
from rbf.normalize import normalizer
import rbf.nodegen
import matplotlib.pyplot as plt
import logging
import modest
logging.basicConfig(level=logging.INFO)

img = Image.open('Lenna.png')
imga = np.array(img,dtype=float)/256.0
c = np.linalg.norm(imga,axis=-1)

N = 50000

t = np.linspace(0,2*np.pi,100)
vert = np.array([0.5+0.5*np.cos(t),0.5+0.5*np.sin(t)]).T
smp = np.array([np.arange(100),np.roll(np.arange(100),-1)]).T

@normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  p = p*512
  p = np.array(p,dtype=int)
  return np.max(c)+0.0001 - c[511-p[:,1],p[:,0]]

modest.tic()
nodes,norms,groups = rbf.nodegen.volume(rho,vert,smp)
print(modest.toc())

plt.plot(nodes[groups==0,0],nodes[groups==0,1],'k.',markersize=3)
plt.plot(nodes[groups==1,0],nodes[groups==1,1],'b.',markersize=5)
plt.show()

