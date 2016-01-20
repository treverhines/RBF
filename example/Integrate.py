#!/usr/bin/env python
import numpy as np
import rbf.integrate
import rbf.halton
#import matplotlib.pyplot as plt
N = 1000
t = np.linspace(0,2*np.pi,N)
vert = np.array([np.cos(t),np.sin(t)],dtype=float).T
smp = np.array([np.arange(N),np.roll(np.arange(N),-1)],dtype=int).T

def rho(p):
  return  1.0 + 10*np.sin(2*p[:,0]) + 10*np.sin(2*p[:,1])

H = rbf.halton.Halton(2)
def rng(N):
  return np.random.random((N,2))

val,err,fmin,fmax = rbf.integrate.rmcint(rho,vert,smp,rng=rng)
#plt.show()
print('integral: %s' % val)
print('estimated error: %s' % err)
print('true error: %s' % (val - np.pi))
print('min value: %s' % fmin)
print('max value: %s' % fmax)







