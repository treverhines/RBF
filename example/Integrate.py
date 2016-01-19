#!/usr/bin/env python
import numpy as np
import rbf.integrate

N = 1000
t = np.linspace(0,2*np.pi,N)
vert = np.array([np.cos(t),np.sin(t)],dtype=float).T
smp = np.array([np.arange(N),np.roll(np.arange(N),-1)],dtype=int).T
tol = 0.01

def rho(p):
  return 1.0 + 0*p[:,0]

def rng(N):
  return np.random.random((N,2))

val,err,fmin,fmax = rbf.integrate.rmcint(rho,vert,smp,rng=rng,tol=tol)
print('integral: %s' % val)
print('estimated error: %s' % err)
print('true error: %s' % (val - np.pi))
print('min value: %s' % fmin)
print('max value: %s' % fmax)







