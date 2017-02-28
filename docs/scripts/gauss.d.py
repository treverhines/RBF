''' 
This script demonstrates how to create a Gaussian process directly 
with the *GaussianProcess* class. In this script we draw samples of 
Brownian motion.
'''
import numpy as np
import matplotlib.pyplot as plt
from rbf.gauss import GaussianProcess
np.random.seed(4)

def brownian_mean(x):
  out = np.zeros(x.shape[0])
  return out

def brownian_cov(x1,x2):
  c = np.min(np.meshgrid(x2[:,0],x1[:,0]),axis=0)
  return c

t = np.linspace(0,1,500)[:,None]
brown = GaussianProcess(brownian_mean,brownian_cov)
samples = [brown.draw_sample(t) for i in range(3)]
mu,sigma = brown(t)

fig,ax = plt.subplots(figsize=(6,5))
ax.grid(True)
ax.plot(t[:,0],mu,'b-')
ax.fill_between(t[:,0],mu+sigma,mu-sigma,color='b',alpha=0.2,edgecolor='none')
for s in samples: ax.plot(t[:,0],s,'k')
ax.set_xlabel('time',fontsize=10)
ax.set_title('Brownian motion')
ax.tick_params(labelsize=10)
fig.tight_layout()
plt.savefig('../figures/gauss.d.png')
plt.show()

