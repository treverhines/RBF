#!/usr/bin/env python
import numpy as np
import rbf.nodegen
from rbf.basis import phs3 as basis
from rbf.integrate import density_normalizer
from rbf.geometry import contains
#from rbf.geometry import enclosure
from rbf.weights import rbf_weight
import rbf.stencil
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs_and_diffs
import numpy as np
import mayavi.mlab
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import logging
import random
import sympy as sp
import mkl
import modest
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

mkl.set_num_threads(1)
logging.basicConfig(level=logging.DEBUG)

def precond1(G):
  out = scipy.sparse.linalg.LinearOperator(G.shape,M.solve)
  return out

def jacobi_preconditioned_solve(G,d):
  N = G.shape[0]
  #G += 10.1*scipy.sparse.eye(N)
  G = G.tocsr()
  #perm = scipy.sparse.csgraph.reverse_cuthill_mckee(G)
  #rev_perm = np.argsort(perm)
  #G = G[perm,:]
  #G = G[:,perm]
  #d = d[perm]
  #G /= 10000.0
  #d /= 10000.0
  #Gar = G.toarray()
  #Gar = np.abs(Gar) + 1e-10
  #plt.imshow(np.log10(Gar))
  #plt.show()
  G = G.astype(np.float64)
  d = d.astype(np.float64)
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data)) # instantiate a matrix
  d = PETSc.Vec().createWithArray(d)
  #soln = scipy.sparse.linalg.spsolve(G,d)
  soln = np.zeros(G.shape[1]) + 0.0
  soln = PETSc.Vec().createWithArray(soln)
  
  #plt.plot(d)
  #plt.show()
  ksp = PETSc.KSP()
  ksp.create()
  ksp.rtol = 1e-20
  ksp.atol = 1e-20
  ksp.max_it = 10000
  ksp.setType('gmres')
  #ksp.setRestart(100)
  ksp.setInitialGuessNonzero(True)
  #ksp.setInitialGuessKnoll(True)
  ksp.setOperators(A)
  ksp.setFromOptions()
  pc = ksp.getPC()
  pc.setType('none')
  pc.setUp()
  ksp.solve(d,soln)
  ksp.view()
  print(ksp.getIterationNumber())
  print(ksp.getResidualNorm())
  print(ksp.getConvergedReason())
  out = np.copy(soln.getArray())
  #out = out[rev_perm]
  return out

#G = scipy.sparse.rand(100,100,0.5)
#m = np.linspace(0,1,100)
#d = G.dot(m)
#G = G.tocsr()
#print(scipy.sparse.linalg.spsolve(G,d))
#print(jacobi_preconditioned_solve(G,d))
#quit()

def _jacobi_preconditioned_solve(G,d):
  #if not scipy.sparse.isspmatrix_csc(G):
  #  G = scipy.sparse.csc_matrix(G)

  # sort G and d using reverse cuthill mckee algorithm

  # form jacobi preconditioner       
  #diag = np.array(G[range(G.shape[0]),range(G.shape[0])])[0]
  #if np.any(diag == 0.0):
  #  raise ValueError(
  #    'matrix cannot be sorted into a diagonal dominant matrix')

  #M = scipy.sparse.diags(1.0/diag,0)
  #out,status = scipy.sparse.linalg.lgmres(G,d,M=M)
  modest.tic('solving')
  #G = G.astype(np.float64) 
  #d= d.astype(np.float64)
  #M = scipy.sparse.linalg.spilu(G.tocsc(),drop_tol=1e-8)
  #pc = scipy.sparse.linalg.LinearOperator(G.shape,M.solve)
  out = scipy.sparse.linalg.spsolve(G,d)
  modest.toc('solving')
  #if status != 0:
  #  raise ValueError(
  #    'lgmres exited with status %s' % status)

  # return to original sorting
  #d = d[rev_perm]
  #out = out[rev_perm]
  #G = G[rev_perm,:]
  #G = G[:,rev_perm]
  return out

# formulate the PDE

x = sp.symbols('x0:3')
n = sp.symbols('n0:3')
u = (sp.Function('u0')(*x),
     sp.Function('u1')(*x),
     sp.Function('u2')(*x))
L = sp.Function('L')(*x)
M = sp.Function('M')(*x)

dim = 3
F = [[u[i].diff(x[j]) for i in range(dim)] for j in range(dim)]
F = sp.Matrix(F)
strain = sp.Rational(1,2)*(F + F.T)
stress = L*sp.eye(dim)*sp.trace(strain) + 2*M*strain
PDEs = [sum(stress[i,j].diff(x[j]) for j in range(dim)) for i in range(dim)]
FreeBCs = [sum(stress[i,j]*n[j] for j in range(dim)) for i in range(dim)]
FixBCs = [u[i] for i in range(dim)]

def lamb(i):
  return 1.0
  #if (nodes[i,1] > 0.05):
  #  return 1.0#2.0
  #if (nodes[i,1] <= 0.05) & (nodes[i,1] >= -0.05):
  #  return 1.0#2.0 - 1.8*(0.05 - nodes[i,1])/0.1
  #if (nodes[i,1] < -0.05):
  #  return 1.0#0.2

def lamb_diffy(i):
  if (nodes[i,1] > 0.05):
    return 0.0
  if (nodes[i,1] <= 0.05) & (nodes[i,1] >= -0.05):
    return 0.0#1.8/0.05
  if (nodes[i,1] < -0.05):
    return 0.0


# norms is an array which is later defined
sym2num = {L:1.0,
           M:1.0,
           sp.Integer(1):1.0,
           sp.Integer(2):2.0,
           L.diff(x[0]):0.0,
           L.diff(x[1]):0.0,
           L.diff(x[2]):0.0,
           M.diff(x[0]):0.0,
           M.diff(x[1]):0.0,
           M.diff(x[2]):0.0,
           n[0]:lambda i:norms[i,0],
           n[1]:lambda i:norms[i,1],
           n[2]:lambda i:norms[i,2]}

DiffOps = [[coeffs_and_diffs(PDEs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FreeBCOps = [[coeffs_and_diffs(FreeBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FixBCOps = [[coeffs_and_diffs(FixBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]

N = 5000
Ns = 20
order = 1

vert = np.array([[0.0,0.0,0.0],
                 [0.0,0.0,1.0],
                 [0.0,1.0,0.0],
                 [0.0,1.0,1.0],
                 [1.0,0.0,0.0],
                 [1.0,0.0,1.0],
                 [1.0,1.0,0.0],
                 [1.0,1.0,1.0]])
smp = np.array([[0,1,4],
                [1,5,4],
                [1,7,5],
                [1,3,7],
                [0,1,3],
                [0,2,3],
                [0,2,6],
                [0,4,6],
                [4,5,7],
                [4,7,6],
                [2,3,7],
                [2,6,7]])

grp = np.ones(len(smp))
# make top a free surface
grp[2] = 2
grp[3] = 2

vert_f = np.array([[0.5001,0.25,0.5],
                   [0.5,0.25,1.0],
                   [0.5,0.75,1.0],
                   [0.5001,0.75,0.5]])
smp_f =  np.array([[0,1,2],
                   [0,2,3]])

@density_normalizer(vert,smp,N)
def rho(p):
  #out = np.zeros(p.shape[0])
  out = 0.00 + 0.5/(1.0 + 1000*np.linalg.norm(p-np.array([0.5,0.5,1.0]),axis=1)**2)
  return out

scale = np.max(vert) - np.min(vert)

# fault nodes
nodes_f,smpid_f,is_edge_f = rbf.nodegen.surface(rho,vert_f,smp_f)

#mayavi.mlab.points3d(nodes_f[:,0],nodes_f[:,1],
#                     nodes_f[:,2],scale_factor=0.005)
#mayavi.mlab.show()

simplex_normals = rbf.geometry.simplex_upward_normals(vert_f,smp_f)

norms_f = simplex_normals[smpid_f]
group_f = np.array(is_edge_f,dtype=int)

nodes_d,smpid_d = rbf.nodegen.volume(rho,vert,smp,
                                     fix_nodes=nodes_f)

simplex_normals = rbf.geometry.simplex_outward_normals(vert,smp)

norms_d = simplex_normals[smpid_d]
norms_d[smpid_d<0] = 0
group_d = np.zeros(nodes_d.shape[0],dtype=int)
group_d[smpid_d>=0] = 1
group_d[smpid_d==2] = 2
group_d[smpid_d==3] = 2

# cut out any fault nodes outside of the domain
is_inside = contains(nodes_f,vert,smp)
nodes_f = nodes_f[is_inside]
norms_f = norms_f[is_inside]
group_f = group_f[is_inside]

nodes_f = nodes_f[group_f==0]
norms_f = norms_f[group_f==0]
group_f = group_f[group_f==0]

slip = np.zeros((dim,len(nodes_f)))
knots_z = np.linspace(0.5,1.5,4)
knots_y = np.linspace(0.25,0.75,4)
import rbf.bspline
basis_no = rbf.bspline.basis_number(knots_z,2)
slip[1,:] = rbf.bspline.bspnd(nodes_f[:,[1,2]],(knots_y,knots_z),(0,0),(2,2))
#slip[1,:] = 1.0

#mayavi.mlab.points3d(nodes_d[:,0],nodes_d[:,1],
#                     nodes_d[:,2],scale_factor=0.005)
#mayavi.mlab.show()

# domain nodes

# split fault nodes into hanging wall and foot wall nodes
nodes_fh = nodes_f + 1e-10*scale*norms_f
nodes_ff = nodes_f - 1e-10*scale*norms_f
norms_fh = norms_f
norms_ff = norms_f


nodes,ix = rbf.nodegen.merge_nodes(interior=nodes_d[group_d==0],
                                   free=nodes_d[group_d==2],
                                   free_ghost=nodes_d[group_d==2],
                                   fixed=nodes_d[group_d==1],
                                   fault_hanging=nodes_fh,
                                   fault_hanging_ghost=nodes_fh,
                                   fault_foot=nodes_ff,
                                   fault_foot_ghost=nodes_ff)


norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_d[group_d==2],
                                   fixed=norms_d[group_d==1],
                                   fault_hanging=norms_fh,
                                   fault_hanging_ghost=norms_fh,
                                   fault_foot=norms_ff,
                                   fault_foot_ghost=norms_ff)


# find the nearest neighbors for the ghost nodes          
s,dx = rbf.stencil.nearest(nodes,nodes,Ns,vert=vert_f,smp=smp_f)

# The closest nodes are going the be the free nodes, which currently      
# are on top of the ghost nodes. find the distance to the next closest    
# node                                                              
dx_next_closest = dx[:,[2]]

# shift the ghost nodes outside           
nodes[ix['free_ghost']] += dx_next_closest[ix['free_ghost']]*norms[ix['free_ghost']]

# move hanging wall ghost nodes towards the foot wall and vice versa                                                  
nodes[ix['fault_hanging_ghost']] -= dx_next_closest[ix['fault_hanging_ghost']]*norms[ix['fault_hanging_ghost']]
nodes[ix['fault_foot_ghost']] += dx_next_closest[ix['fault_foot_ghost']]*norms[ix['fault_foot_ghost']]

#for k,v in ix.iteritems():
#  mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,opacity=0.5,color=(1,0,0))
#  mayavi.mlab.triangular_mesh(vert_f[:,0],vert_f[:,1],vert_f[:,2],smp_f,opacity=0.5,color=(1,0,0))
#  mayavi.mlab.points3d(nodes[v,0],nodes[v,1],nodes[v,2],scale_factor=0.01)
#  print(k)
#  mayavi.mlab.show()

N = len(nodes)

modest.tic('buildingG')
G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float64) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N) for mi in range(dim)]
# This can be parallelized!!!!
for di in range(dim):
  for mi in range(dim):
    for i in ix['interior']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],i),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[i]] = w

    for i in ix['free']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[i]] = w

    for i in ix['fixed']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(FixBCOps[di][mi],i),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[i]] = w

    # treat fault nodes as free nodes and the later reorganize the G                                                  
    # matrix so that the fault nodes are forces to be equal                                                           
    for i in ix['fault_hanging']+ix['fault_foot']:
      w = rbf_weight(nodes[i],
                 nodes[s[i]],
                 evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                 order=order,
                 basis=basis)

      G[di][mi][i,s[i]] = w

    # use the ghost node rows to enforce the free boundary conditions                                                 
    # at the boundary nodes                                                                                           
    for itr,i in enumerate(ix['free_ghost']):
      j = ix['free'][itr]
      w = rbf_weight(nodes[j],
                     nodes[s[j]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[j]] = w

    for itr,i in enumerate(ix['fault_hanging_ghost']):
      j = ix['fault_hanging'][itr]
      w = rbf_weight(nodes[j],
                     nodes[s[j]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[j]] = w

    for itr,i in enumerate(ix['fault_foot_ghost']):
      j = ix['fault_foot'][itr]
      w = rbf_weight(nodes[j],
                     nodes[s[j]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                     order=order,
                     basis=basis)

      G[di][mi][i,s[j]] = w

for di in range(dim):
  for mi in range(dim):
    G[di][mi].tocsc()
    data[di] = (data[di] +
                G[di][mi][:,ix['fault_foot']].dot(slip[mi,:]) +
                G[di][mi][:,ix['fault_hanging']].dot(-slip[mi,:]))

    G[di][mi][:,ix['fault_foot']] = G[di][mi][:,ix['fault_foot']] + G[di][mi][:,ix['fault_hanging']]
    G[di][mi][:,ix['fault_hanging']] = 0
    G[di][mi][ix['fault_foot'],:] = G[di][mi][ix['fault_foot'],:] - G[di][mi][ix['fault_hanging'],:]
    G[di][mi][ix['fault_hanging'],:] = 0

  G[di][di][ix['fault_hanging'],ix['fault_hanging']] = 1
  data[di][ix['fault_foot']] = data[di][ix['fault_foot']] - data[di][ix['fault_hanging']]
  data[di][ix['fault_hanging']] = 0


G = [scipy.sparse.hstack(G[i]) for i in range(dim)]
G = scipy.sparse.vstack(G)
data = np.concatenate(data)
G = G.tocsc()
modest.toc('buildingG')

out = jacobi_preconditioned_solve(G,data)
out = np.reshape(out,(dim,N))
out[:,ix['fault_foot']] = out[:,ix['fault_foot']] - slip
out[:,ix['fault_hanging']] = out[:,ix['fault_foot']] + 2*slip
out = out.T

idx = ix['fault_foot'] + ix['fault_hanging'] + ix['interior'] + ix['fixed'] + ix['free']

import matplotlib.pyplot as plt
plt.quiver(nodes[ix['free'],0],nodes[ix['free'],1],out[ix['free'],0],out[ix['free'],1])
plt.show()

#mayavi.mlab.quiver3d(nodes[idx,0],nodes[idx,1],nodes[idx,2],out[idx,0],out[idx,1],out[idx,2],mode='arrow',color=(0,1,0))


#mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,opacity=0.2,color=(1,1,1))
#mayavi.mlab.triangular_mesh(vert_f[:,0],vert_f[:,1],vert_f[:,2],smp_f,opacity=0.2,color=(1,0,0))

#mayavi.mlab.show()
logging.basicConfig(level=logging.INFO)





