#!/usr/bin/env python
import numpy as np
import rbf.fd
import rbf.basis
import sympy 
import matplotlib.pyplot as plt


diff = (3,0)
x,y = sympy.symbols('x,y')
r = sympy.sqrt(x**2 + y**2)
u_sym = 1/(1 + r**2)
udiff_sym = u_sym.diff(x,x,x)
u = sympy.lambdify((x,y),u_sym,'numpy')
udiff = sympy.lambdify((x,y),udiff_sym,'numpy')


# number of nodes
N = 20
# node spread
sigma = 0.1

center = np.random.normal(0.0,1.0,2)

fig,ax = plt.subplots()

udiff_true = udiff(center[0],center[1])

N = 10
P = 3
for i,v in enumerate(np.linspace(0.0,-8.0,100)):
  sigma = 10**(v)
  err = 0.0
  for i in range(1000): 
    nodes = center + np.random.normal(0.0,sigma,(N,2))
    obs = u(nodes[:,0],nodes[:,1])
    w = rbf.fd.weights(center,nodes,diff,order=P)
    err += abs(udiff_true - w.dot(obs))

  err /= 1000
  ax.loglog(sigma,err,'ko')

N = 15
P = 3
for i,v in enumerate(np.linspace(0.0,-8.0,100)):
  sigma = 10**(v)
  err = 0.0
  for i in range(100): 
    nodes = center + np.random.normal(0.0,sigma,(N,2))
    obs = u(nodes[:,0],nodes[:,1])
    w = rbf.fd.weights(center,nodes,diff,order=P)
    err += abs(udiff_true - w.dot(obs))

  err /= 100
  ax.loglog(sigma,err,'bo')

N = 20
P = 3
for i,v in enumerate(np.linspace(0.0,-8.0,100)):
  sigma = 10**(v)
  err = 0.0
  for i in range(100): 
    nodes = center + np.random.normal(0.0,sigma,(N,2))
    obs = u(nodes[:,0],nodes[:,1])
    w = rbf.fd.weights(center,nodes,diff,order=P)
    err += abs(udiff_true - w.dot(obs))

  err /= 100
  ax.loglog(sigma,err,'ro')

N = 25
P = 3
for i,v in enumerate(np.linspace(0.0,-8.0,100)):
  sigma = 10**(v)
  err = 0.0
  for i in range(100): 
    nodes = center + np.random.normal(0.0,sigma,(N,2))
    obs = u(nodes[:,0],nodes[:,1])
    w = rbf.fd.weights(center,nodes,diff,order=P)
    err += abs(udiff_true - w.dot(obs))

  err /= 100
  ax.loglog(sigma,err,'mo')

N = 30
P = 3
for i,v in enumerate(np.linspace(0.0,-8.0,100)):
  sigma = 10**(v)
  err = 0.0
  for i in range(100): 
    nodes = center + np.random.normal(0.0,sigma,(N,2))
    obs = u(nodes[:,0],nodes[:,1])
    w = rbf.fd.weights(center,nodes,diff,order=P)
    err += abs(udiff_true - w.dot(obs))

  err /= 100
  ax.loglog(sigma,err,'co')
      

plt.show()







