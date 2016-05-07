#!/usr/bin/env python

# This is a demonstration for using PETSc to solve a 1D ODE.
# The ODE we are solving is
# 
#   d^2u/dx^2 = 0
#
# with BCs:
#
#   u(x=0.0) = 2.0
#   u(x=1.0) = 3.0
#
# A handful of packages are necessary for this problem.  They are 
# PETSc, PETSc4py, and mpi4py.  These packages can be installed 
# through pip.   

# It seems that the pip repository does not have the 
# most up to date versions of PETSc and PETSc4py and you may need to 
# download and install PETSc from https://www.mcs.anl.gov/petsc.  Once 
# petsc is installed, set the PETSC_PATH and PETSC_ARCH environment 
# variables and install PETSc4py by pointing pip to the bitbucket 
# repository with the command

#   $ pip install https://bitbucket.org/petsc/petsc4py/get/master.tar.gz
#
# to run this script in parallel:
#
#   mpiexec -n {cores} 1DPETSc.py

import numpy as np 
import rbf.basis 
import rbf.weights 
import rbf.halton 
import rbf.stencil 
import sys
import scipy.sparse
import petsc4py 
petsc4py.init(sys.argv) 
from petsc4py import PETSc 
import matplotlib.pyplot as plt 
import modest 
import logging 
logging.basicConfig(level=logging.INFO)

def petsc_solve(G,d):
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data)) # instantiate a matrix
  d = PETSc.Vec().createWithArray(d)
  soln = np.zeros(G.shape[1])
  soln = PETSc.Vec().createWithArray(soln)

  ksp = PETSc.KSP()
  ksp.create()
  #ksp.setType('gmres')
  #ksp.getPC().setType('jacobi')
  ksp.setOperators(A)
  #ksp.rtol = 1e-5
  #ksp.max_it = 10000 
  ksp.solve(d,soln)
  return soln.getArray()


comm = PETSc.COMM_WORLD

# print the rank of the current process
rank = comm.Get_rank()
print('running process %s' % rank) 

# For demonstration purposes, wait for all processes to reach this point
comm.Barrier()

# run this code only for the first processes
if rank == 0:
  # get the total number of processes
  total_procs = comm.Get_size()
  print('Total running processes: %s' % total_procs)

N = 1000 # number of nodes
S = 10 # nodes per stencil

# create nodes
modest.tic('building')
x = rbf.halton.halton(N)[:,0]
x = np.sort(x)
x = x[:,None]
stencil,dx = rbf.stencil.nearest(x,x,S)
# PETSc requires the stencil to be a 32 bit integer type
stencil = stencil.astype(np.int32)

data = 1.0 + np.zeros(N)
data[0] = 2.0
data[N-1] = 3.0

soln = np.zeros(N)

#G = scipy.sparse.csr_matrix()
weight = np.zeros((N,S))
for i,s in enumerate(stencil):
  if (i == 0) | (i==(N-1)):  
    # note that the first time rbf_weight is called it compiles sympy 
    # code to cython code. This produces about a 1 second overhead
    weight[i,:] = rbf.weights.rbf_weight(x[i],x[s],(0,))
  else:
    weight[i,:] = rbf.weights.rbf_weight(x[i],x[s],(2,))

rows = np.repeat(np.arange(N),S)
cols = stencil.flatten()
weight = weight.flatten()
modest.toc('building')

G = scipy.sparse.csr_matrix((weight,(rows,cols)))
modest.tic('scipy solve')
soln2 = scipy.sparse.linalg.spsolve(G,data)
modest.toc('scipy solve')
#print(soln)


modest.tic('petsc')
soln1 = petsc_solve(G,data)
modest.toc('petsc')


#ksp.setInitialGuessNonzero(b
#ksp.setFromOptions()
#ksp.setUp()
#pc = ksp.getPC()
#pc.setType('bjacobi')
#print(ksp.getName())
#print(help(ksp))
#ksp.setUp()
#print('preconditioning with', pc)
#print 'Solving with:', ksp.getType()

# Solve!
modest.tic('petsc solve')
#print(ksp.getConvergedReason())
#print(ksp.getIterationNumber())
modest.toc('petsc solve')
#print(soln.getArray())
#plt.plot(x,pout.getArray())
#plt.show()
plt.plot(x,soln1,'k')
plt.plot(x,soln2,'b')
plt.show()
modest.summary()








