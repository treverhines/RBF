#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.smooth
import rbf.domain
from scipy.signal import periodogram
import time

N = 1000
t = np.linspace(0.0,10.0,N)[:,None]
u = 0*np.sin(t[:,0]) + np.random.normal(0.0,1.0,N)
sigma = np.ones(N)

a = time.time()
u1,cov = rbf.smooth.smooth(t,u,sigma=sigma,cutoff=1.0,fill='interpolate',return_covariance=False)
print((time.time() - a)*1000)

