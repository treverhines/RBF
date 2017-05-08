''' 
This script demonstrates the power the GaussianProcess class now that
it can handle sparse covariances!
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import rbf.gauss
import rbf
import logging
logging.basicConfig(level=logging.DEBUG)
np.random.seed(1)

t = np.arange(0.0,10.0,1.0/365.25)[:,None]
n = len(t)

titp = np.linspace(4.0,7.0,n)[:,None]
s = 0.5*np.ones(n)
d = np.sin(t[:,0]) + np.random.normal(0.0,s)

def basis(x):
  return np.array([np.sin(x[:,0]),np.cos(x[:,0])]).T
  
  
gp1  = rbf.gauss.gppoly(1)
gp1 += rbf.gauss.gpiso(rbf.basis.spwen11,(0.0,1.0,0.1))


gp2  = rbf.gauss.gppoly(1)
gp2 += rbf.gauss.gpiso(rbf.basis.wen11,(0.0,1.0,0.1))

fig,ax = plt.subplots()
gpc = gp1.condition(t,d,s)
#u,us = gpc.meansd(titp,chunk_size=1000)
u = gpc.mean(titp)
print(u.shape)
ax.plot(titp[:,0],u,'b-')
#ax.fill_between(titp[:,0],u-us,u+us,color='b',alpha=0.2)

gpc = gp2.condition(t,d,s)
#u,us = gpc.meansd(titp,chunk_size=1000)
print(titp.shape)
u = gpc.mean(titp)
print(u.shape)
ax.plot(titp[:,0],u,'r-')
#ax.fill_between(titp[:,0],u-us,u+us,color='r',alpha=0.2)

ax.plot(t[:,0],d,'k.',alpha=0.1)

plt.show()
