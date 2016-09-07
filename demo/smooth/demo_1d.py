#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.smooth
import rbf.halton
import rbf.domain
import rbf.basis
import rbf.nodes
from scipy.signal import periodogram
import scipy.sparse.linalg as spla
import time
import cProfile
import logging
rbf.basis.set_sym_to_num('cython')
logging.basicConfig(level=logging.DEBUG)

N = 10000
vert = np.array([[0.0,0.0],
                 [1.0,0.0],
                 [1.0,1.0],
                 [0.0,1.0]])
smp = np.array([[0,1],
                [1,2],                 
                [2,3],    
                [3,0]])
x,sid = rbf.nodes.menodes(N,vert,smp,itr=1)

u = np.random.normal(0.0,1.0,N) + np.sin(2*np.pi*x[:,0]) + np.sin(10*np.pi*x[:,0]) + 10*np.cos(20*np.pi*x[:,1])
sigma = np.ones(N)
a = time.time()
post_mean,post_std = rbf.smooth.smooth(x,u,sigma=sigma,cutoff=2.0,fill='none')
print(time.time() - a)
plt.figure(1)
plt.tripcolor(x[:,0],x[:,1],u)
plt.colorbar()
plt.figure(2)
plt.tripcolor(x[:,0],x[:,1],post_mean,vmin=-1.2,vmax=1.2)
plt.colorbar()
plt.figure(3)
plt.tripcolor(x[:,0],x[:,1],post_std)
plt.colorbar()
#plt.plot(x[:,0],x[:,1],'k.')
plt.show()



