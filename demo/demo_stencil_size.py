#!/usr/bin/env python
# This script demonstrates how the stencil size affects the accuracy 
# of the RBF-FD approximation
import numpy as np
import sympy
import rbf.basis
import rbf.fd
import rbf.nodes
import rbf.halton
import matplotlib.pyplot as plt
import logging
import rbf.domain
logging.basicConfig(level=logging.DEBUG)

#### 2D TEST
#####################################################################

# RBF-FD free parameters
basis = rbf.basis.phs3
order = None

# make circular domain
N = 500
#vert,smp = rbf.domain.sphere()
#nodes,sid = rbf.nodes.make_nodes(N,vert,smp,itr=100)
# make test function 
# make symbolic test function
x,y = sympy.symbols('x,y')
f_sym = sympy.sin(2*x)*sympy.cos(2*y)
diff = (4,0)
fdiff_sym = f_sym.diff(x,x,x,x)
# make numerical test function
f = sympy.lambdify((x,y),f_sym,'numpy')
fdiff = sympy.lambdify((x,y),fdiff_sym,'numpy')

itr = 10
#all_nodes = np.random.random((itr,N,3))
H = rbf.halton.Halton(2)
all_nodes = [H(N) for i in range(itr)]
def compute_max_err(S):
  out = []
  for i in range(itr):
    nodes = all_nodes[i]
    u = f(nodes[:,0],nodes[:,1])
    udiff = fdiff(nodes[:,0],nodes[:,1])
    D = rbf.fd.diff_matrix(nodes,diff,N=S,basis=basis,order=order)
    udiff_est = D.dot(u)
    max_err = np.max(np.abs(udiff - udiff_est))
    out += [max_err]

  return out

#D = rbf.fd.diff_matrix(nodes,diff,N=10,basis=basis,order=order)
#udiff_est = D.dot(u)
#err = np.abs(udiff - udiff_est)
#plt.figure(1)
#p = plt.scatter(nodes[:,0],nodes[:,1],s=100,c=np.log10(err),edgecolor='none')
#plt.colorbar(p)

plt.figure(2)
#sizes = np.arange(3,20)
#errs = [compute_max_err(s) for s in sizes]
#plt.semilogy(sizes,errs,'k-')

#order = 0
sizes = np.arange(15,50,1)
errs = np.array([compute_max_err(s) for s in sizes])
means = np.mean(errs,axis=1)
std = np.std(errs,axis=1)/np.sqrt(itr)
plt.errorbar(sizes,means,std)
''' 
order = 1
sizes = np.arange(4,50,1)
errs = [compute_max_err(s) for s in sizes]
plt.semilogy(sizes,errs,'g-')

order = 2
sizes = np.arange(6,50,1)
errs = [compute_max_err(s) for s in sizes]
plt.semilogy(sizes,errs,'b-')

order = 3
sizes = np.arange(10,50,1)
errs = [compute_max_err(s) for s in sizes]
plt.semilogy(sizes,errs,'m-')

order = 4
sizes = np.arange(15,50,1)
errs = [compute_max_err(s) for s in sizes]
plt.semilogy(sizes,errs,'r-')

order = 5
sizes = np.arange(21,50,1)
errs = [compute_max_err(s) for s in sizes]
plt.semilogy(sizes,errs,'c-')

'''
plt.show()

