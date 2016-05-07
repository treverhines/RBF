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
import petsc4py 
petsc4py.init(sys.argv) 
from petsc4py import PETSc 
import matplotlib.pyplot as plt 
import modest 
import logging 
logging.basicConfig(level=logging.INFO)

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

N = 10 # number of nodes
S = 4 # nodes per stencil

# create nodes
x = rbf.halton.halton(N)[:,0]
x = np.sort(x)
x = x[:,None]
stencil,dx = rbf.stencil.nearest(x,x,S)

# PETSc requires the stencil to be a 32 bit integer type
stencil = stencil.astype(np.int32)

data = np.zeros(N)
data[0] = 2.0
data[N-1] = 3.0

A = PETSc.Mat() # instantiate a matrix
A.create(comm) # share this matrix with COMM_WORLD 
A.setSizes([N,N]) # set the global dimensions of the array
A.setType('aij') # set it to the standard sparse format
A.setPreallocationNNZ(S) # set the Number of Non-Zero entries per row
Istart, Iend = A.getOwnershipRange() # Find out which rows were allocated to this process
modest.tic('building new G')
# loop over the unique rows for each process
for i in range(Istart,Iend):
  s = stencil[i] 
  if (i == 0) | (i==(N-1)):  
    # note that the first time rbf_weight is called it compiles sympy 
    # code to cython code. This produces about a 1 second overhead
    data = rbf.weights.rbf_weight(x[i],x[s],(0,))

  else:
    data = rbf.weights.rbf_weight(x[i],x[s],(2,))

  A.setValues([i],s,data)

#print(help(A))
modest.toc('building new G')

A.assemblyBegin()
A.assemblyEnd()
#A.view()

a = PETSc.Vec()
a.create(comm)
a.setSizes(N)
a.setType('mpi')
a.set(1.0)
b = PETSc.Vec()
b.create(comm)
b.setSizes(N)
b.setType('mpi')
b.set(1.0)
print(a.getArray())
#pdata.setSizes([N])
#pdata.setFromOptions()
#pdata.createSeq(N)

#for i in range(N):
#  pdata.setValue(i,data[i])

#pout = PETSc.Vec(comm).createSeq(N)

#c,b = A.getVecs()
#c.set(1.0)
#b.set(0.0)
print(a.getType())
print(A.getType())
#quit()
#a.setValues(0,2.0)
#a.setValues(N,2.0)

ksp = PETSc.KSP()
ksp.create()
#ksp.setType('gmres')
#ksp.getPC().setType('icc')
ksp.setOperators(A)
#ksp.setInitialGuessNonzero(b)
ksp.setFromOptions()
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
ksp.solve(a,b)
modest.toc('petsc solve')
print(b.getArray())
#plt.plot(x,pout.getArray())
#plt.show()

modest.summary()








