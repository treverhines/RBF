''' 
This script demonstrates how the *GaussianProcess* class is smart
enough to take advantage of sparse covariance function. The *gppoly*
and *gpbfci* constructors returns a *GaussianProcess* with a sparse
(entirely zero) covariance function. We can also construct a
*GaussianProcess* with a sparse covariance matrix by calling *gpiso*
and a *SparseRBF* instance. 
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import rbf.gauss
import logging
logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

# create synthetic data
n = 10000
y = np.linspace(-20.0,20.0,n) # observation points
sigma = 0.5*np.ones(n)
d = np.exp(-0.3*np.abs(y))*np.sin(y) + np.random.normal(0.0,sigma)
# evaluate the output at a subset of the observation points
x = np.linspace(-20.0,20.0,1000) # interpolation points
u_true = np.exp(-0.3*np.abs(x))*np.sin(x)  # true signal
# create a sparse GP
gp = rbf.gauss.gpiso(rbf.basis.spwen12,(0.0,1.0,4.0)) 
# condition with the observations
gpc = gp.condition(y[:,None],d,sigma)
# find the mean and std of the conditioned GP. Chunk size controls the
# trade off between speed and memory consumption. It should be tuned
# by the user.
u,us = gpc.meansd(x[:,None],chunk_size=1000)

fig,ax = plt.subplots()
ax.plot(x,u_true,'k-',label='true signal')
ax.plot(y,d,'k.',alpha=0.1,mec='none',label='observations')
ax.plot(x,u,'b-',label='post. mean')
ax.fill_between(x,u-us,u+us,color='b',alpha=0.2,label='post. std. dev.')
ax.set_xlim((-20.0,20.0))
ax.set_ylim((-2.0,2.0))
ax.legend(loc=2,fontsize=10)
#plt.show()
