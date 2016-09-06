#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.smooth
import rbf.domain
import rbf.basis
from scipy.signal import periodogram
import scipy.sparse.linalg as spla
import time
import cProfile
rbf.basis.set_sym_to_num('cython')

N = 500
t = np.linspace(0.0,10.0,N)[:,None]
u = 0*np.sin(t[:,0]) + np.random.normal(0.0,1.0,N)
sigma = np.ones(N)
#sigma[((t[:,0]>2.0) & (t[:,0]<2.5))] = np.inf
#sigma = np.random.uniform(0.5,10.0,N)
#sigma = np.random.uniform(0.5,2.0,N)
#sigma = np.random.uniform(0.5,2.0,N)

a = time.time()
#for i in range(100):
#  print(i)
post_mean,post_std = rbf.smooth.smooth(t,u,sigma=sigma,order=2,cutoff=1.0,samples=400)
cProfile.run('for i in range(100): rbf.smooth.smooth(t,u,sigma=sigma,order=2,cutoff=1.0,samples=100)')

print((time.time() - a))




