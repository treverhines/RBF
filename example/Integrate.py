#!/usr/bin/env python
import numpy as np
import rbf.integrate
import rbf.halton
import matplotlib.pyplot as plt
import modest
import logging
logging.basicConfig(level=logging.INFO)
N = 100
t = np.linspace(0,2*np.pi,N)
vert = np.array([np.cos(t),np.sin(t)],dtype=float).T
smp = np.array([np.arange(N),np.roll(np.arange(N),-1)],dtype=int).T

def rho(p):
  return  0.0 + 1*p[:,0]

H = rbf.halton.Halton(2)
def rng(N):
  return np.random.random((N,2))

modest.tic()
val,err,fmin,fmax = rbf.integrate.rmcint(rho,vert,smp,tol=0.001)

print(modest.toc())
#vals = [rbf.integrate.rmcint(rho,vert,smp,rng=rng)[0] for i in range(500)]
#plt.hist(vals,20)
print('integral: %s' % val)
print('estimated error: %s' % err)
#print('true error: %s' % np.std(vals))
print('min value: %s' % fmin)
print('max value: %s' % fmax)
plt.show()






