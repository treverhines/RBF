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

def seasonals(x):
  return np.array([np.sin(2*np.pi*x[:,0]),
                   np.cos(2*np.pi*x[:,0])]).T
                   
# observation times
times = np.linspace(0.0,10.0,1000)[:,None]
# GaussianProcess describing the underlying signal
signal_gp   = gppoly(1)
signal_gp  += gpse((0.0,0.5,2.0))
# GaussianProcess describing continuous noise 
noise_gp    = gpexp((0.0,0.05,0.2))
noise_gp   += gpbfc(seasonals,[0.0,0.0],[0.2,0.1])
# standard deviation for discrete noise
noise_disc  = 0.1*np.ones(times.shape[0]) 
# underlying signal we want to recover
true = signal_gp.sample(times,c=[1.0,-0.2])
# true signal plus noise
obs  = (true + 
        noise_gp.sample(times) + 
        np.random.normal(0.0,noise_disc))
# discrete covariance matrix for the observations
sigma = (noise_gp.covariance(times,times) + 
         np.diag(noise_disc**2))
# condition the signal GaussianProcess with the observations
cond_gp = signal_gp | (times,obs,sigma)
# evaluate the mean and std. dev.
pred,sigma = cond_gp(times)

fig,ax = plt.subplots()
ax.plot(times[:,0],obs,'C0.')
ax.plot(times[:,0],true,'C1-')
ax.plot(times[:,0],pred,'C2-')
ax.fill_between(times[:,0],pred-sigma,pred+sigma,color='C2',alpha=0.2)

plt.show()
               
               
                  




