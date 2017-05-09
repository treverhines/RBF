''' 
This script demonstrates how the *GaussianProcess* class is smart
enough to take advantage of sparse covariance function. The *gppoly*
and *gpbfci* constructors returns a *GaussianProcess* with a sparse
(entirely zero) covariance function. Conditioning the resulting
*GaussianProcess* is equivalent to linear regression of the basis
functions with the observations. Since we are taking advantage of the
sparsity, conditioning the *GaussianProcess* is also just as
computationally efficient as linear regression.
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import rbf.gauss
import logging
logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

# create synthetic data
n = 100000
t = np.linspace(0.0,10.0,n)[:,None]
s = 0.5*np.ones(n)
d = 1.0 + 0.2*t[:,0] - 0.05*t[:,0]**2 + np.random.normal(0.0,s)
# evaluate the output at a subset of the observation points
x = t[::1000]

# create a GP with polynomial basis functions  
gp = rbf.gauss.gppoly(2)
# condition with the observations
gpc = gp.condition(t,d,s)
# find the mean and std of the conditioned GP. Chunk size controls the
# trade off between speed and memory consumption. It should be tuned
# by the user.
u,us = gpc.meansd(x,chunk_size=1000)

fig,ax = plt.subplots()
ax.plot(t[:,0],d,'k.',alpha=0.05,mec='none')
ax.plot(x[:,0],u,'b-')
ax.fill_between(x[:,0],u-us,u+us,color='b',alpha=0.2)


plt.show()
