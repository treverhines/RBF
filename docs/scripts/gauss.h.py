'''  
This scripts demonstrates how to perform Gaussian process regression
on a timeseries with a complicated noise model. The noise is treated
as a combination of white noise, exponential noise, and seasonal
terms. The signal is treated as a first order polynomial plus a
squared exponential.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.gauss import gpse,gpexp,gppoly,gpbfc
#from pygeons.filter.gpr import gpr,gpfogm,gpseasonal
import logging
logging.basicConfig(level=logging.DEBUG)

def seasonals(x):
  return np.array([np.sin(2*np.pi*x[:,0]),
                   np.cos(2*np.pi*x[:,0])]).T
                   
# observation times
dt = 1.0
times = np.arange(0.0,500.0,dt)[:,None]
# GaussianProcess describing the underlying signal
signal_gp   = gppoly(1)
signal_gp  += gpse((0.0,2.0**2,50.0))
# GaussianProcess describing continuous noise 
noise_gp    = gpfogm(0.5,10.0)
noise_gp   += gpseasonal(True,True)
# standard deviation for discrete noise
noise_sigma  = 0.1*np.ones(times.shape[0]) 
# underlying signal we want to recover
true = signal_gp.sample(times,c=[1.0,0.5])
# true signal plus noise
obs  = (true + 
        noise_gp.sample(times,c=[1.0,1.0,1.0,1.0]) + 
        np.random.normal(0.0,noise_sigma))

pred,sigma = gpr(times,obs,noise_sigma,(2.0,50.0),
                 fogm_params=(0.5,10.0),                
                 annual=True,semiannual=True,tol=4.0)

diff,diff_sigma = gpr(times,obs,noise_sigma,(2.0,50.0),diff=(1,),
                      annual=True,semiannual=True,tol=4.0)
## discrete covariance matrix for the observations
#sigma = (noise_gp.covariance(times,times) + 
#         np.diag(noise_disc**2))
## condition the signal GaussianProcess with the observations
#cond_gp = signal_gp | (times,obs,sigma)
## evaluate the mean and std. dev.
#pred,sigma = cond_gp(times)
#
fig,ax = plt.subplots()
ax.plot(times[:,0],obs,'C0.')
ax.plot(times[:,0],true,'C1-')
ax.plot(times[:,0],pred,'C2-')
ax.fill_between(times[:,0],pred-sigma,pred+sigma,color='C2',alpha=0.2)

fig,ax = plt.subplots()
ax.plot(times[1:,0],np.diff(true)/np.diff(times[:,0]),'C1-')
ax.plot(times[:,0],diff,'C2-')
ax.fill_between(times[:,0],diff-diff_sigma,diff+diff_sigma,color='C2',alpha=0.2)

plt.show()
               
               
                  




