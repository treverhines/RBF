#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.smooth
import rbf.domain
import rbf.basis
from scipy.signal import periodogram
import scipy.sparse.linalg as spla
import time
rbf.basis.set_sym_to_num('cython')

N = 10000
t = np.linspace(0.0,10.0,N)[:,None]
u = 0*np.sin(t[:,0]) + np.random.normal(0.0,1.0,N)
sigma = np.ones(N)
sigma[((t[:,0]>2.0) & (t[:,0]<2.5))] = np.inf
#sigma = np.random.uniform(0.5,10.0,N)
#sigma = np.random.uniform(0.5,2.0,N)
#sigma = np.random.uniform(0.5,2.0,N)

post_mean1,post_std1 = rbf.smooth.smooth(t,u,sigma=sigma,order=2,cutoff=1.0,itr=1000,fill='none')
post_mean2,post_std2 = rbf.smooth.smooth(t,u,sigma=sigma,order=2,cutoff=1.0,itr=1000)
plt.plot(t,post_mean1,'b-')
plt.fill_between(t[:,0],post_mean1-post_std1,post_mean1+post_std1,color='b',alpha=0.2)
plt.plot(t,post_mean2,'b-')
plt.fill_between(t[:,0],post_mean2-post_std2,post_mean2+post_std2,color='b',alpha=0.2)
plt.show()




