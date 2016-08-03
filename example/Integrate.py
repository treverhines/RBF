#!/usr/bin/env python
#
# This script uses the three Monte Carlo integration functions within
# rbf.integrate to integrate a the function rho over a unit circle
import numpy as np
import rbf.integrate
import rbf.halton
import matplotlib.pyplot as plt

# number of segments used to make the circle
N = 1000
t = np.linspace(0,2*np.pi,N)

# vertices and simplices defining the circle
vert = np.array([np.cos(t),np.sin(t)],dtype=float).T
smp = np.array([np.arange(N),np.roll(np.arange(N),-1)],dtype=int).T

# function being integrated
def rho(p):
  return  1.0 + 0.0*p[:,0]


val,err,fmin,fmax = rbf.integrate.mcint(rho,vert,smp)
print('integral: %s' % val)
print('estimated error: %s' % err)
print('min value: %s' % fmin)
print('max value: %s' % fmax)

val,err,fmin,fmax = rbf.integrate.mcint2(rho,vert,smp)
print('integral: %s' % val)
print('estimated error: %s' % err)
print('min value: %s' % fmin)
print('max value: %s' % fmax)

val,err,fmin,fmax = rbf.integrate.rmcint(rho,vert,smp)
print('integral: %s' % val)
print('estimated error: %s' % err)
print('min value: %s' % fmin)
print('max value: %s' % fmax)







