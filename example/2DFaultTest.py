#!/usr/bin/env python
import numpy as np
import rbf.nodegen
import modest
from rbf.basis import phs3 as basis
from rbf.integrate import density_normalizer
from rbf.geometry import contains
from rbf.weights import rbf_weight
import rbf.stencil
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs_and_diffs
import numpy as np
import rbf.bspline
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.sparse
import scipy.sparse.linalg
import logging
import sympy as sp
import multiprocessing as mp
import mkl
import myplot.cm
logging.basicConfig(level=logging.INFO)
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc


def solver(G,d,init_guess=None):
  # TO VIEW THE RESIDUALS FOR EACH ITERATION CALL THIS SCRIPT WITH
  #
  #  python 2DFaultTest.py -ksp_monitor_true_residual
  #
  N = G.shape[0]
  #G += 10.1*scipy.sparse.eye(N)
  G = G.tocsr()
  #print(np.linalg.cond(G.toarray()))
  G = G.astype(np.float64)
  d = d.astype(np.float64)
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data)) # instantiate a matrix
  d = PETSc.Vec().createWithArray(d)
  #soln = scipy.sparse.linalg.spsolve(G,d)
  if init_guess is None:
    soln = np.zeros(G.shape[1]) + 0.0
  else:
    soln = init_guess
  
  #soln[ix['fault_foot']] = -1.0
  #soln[ix['fault_hanging']] = 1.0
  soln = PETSc.Vec().createWithArray(soln)

  #plt.plot(d)
  #plt.show()
  ksp = PETSc.KSP()
  ksp.create()
  #ksp.monitor_true_residual = True
  ksp.rtol = 1e-10
  ksp.atol = 1e-5
  ksp.max_it = 10000
  ksp.setType('gmres')
  #ksp.setRestart(100)
  #ksp.setInitialGuessNonzero(True)
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

def _solver(G,d):
  if not scipy.sparse.isspmatrix_csc(G):
    G = scipy.sparse.csc_matrix(G)

  # sort G and d using reverse cuthill mckee algorithm
  perm = scipy.sparse.csgraph.reverse_cuthill_mckee(G)  
  rev_perm = np.argsort(perm)
  G = G[perm,:]
  G = G[:,perm]
  d = d[perm]

  # it is not clear whether sorting the matrix helps much
  modest.tic('solving')
  #print(np.linalg.cond(G.toarray())) 
  out = scipy.sparse.linalg.spsolve(G,d,use_umfpack=False)
  print(modest.toc('solving'))

  # return to original sorting
  d = d[rev_perm]
  out = out[rev_perm]
  G = G[rev_perm,:]
  G = G[:,rev_perm]
  return out   


# formulate the PDE
x = sp.symbols('x0:2')
n = sp.symbols('n0:2')
u = (sp.Function('u0')(*x),
     sp.Function('u1')(*x))
L = sp.Function('L')(*x)
M = sp.Function('M')(*x)

dim = 2
F = [[u[i].diff(x[j]) for i in range(dim)] for j in range(dim)]
F = sp.Matrix(F)
strain = sp.Rational(1,2)*(F + F.T)
stress = L*sp.eye(dim)*sp.trace(strain) + 2*M*strain
PDEs = [sum(stress[i,j].diff(x[j]) for j in range(dim)) for i in range(dim)]
FreeBCs = [sum(stress[i,j]*n[j] for j in range(dim)) for i in range(dim)]
FixBCs = [u[i] for i in range(dim)]

def lamb(i):
  if (nodes[i,1] > 0.05):
    return 1.0#2.0
  if (nodes[i,1] <= 0.05) & (nodes[i,1] >= -0.05):
    return 1.0#2.0 - 1.8*(0.05 - nodes[i,1])/0.1
  if (nodes[i,1] < -0.05):
    return 1.0#0.2

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
           L.diff(x[1]):lamb_diffy,
           M.diff(x[0]):0.0,
           M.diff(x[1]):lamb_diffy,
           n[0]:lambda i:norms[i,0],
           n[1]:lambda i:norms[i,1]}

DiffOps = [[coeffs_and_diffs(PDEs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FreeBCOps = [[coeffs_and_diffs(FreeBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FixBCOps = [[coeffs_and_diffs(FixBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]

# The number of nodes needed depends on how sharp slip varies on the fault
N = 1000
Ns = 8
# It seems like using the maximum polynomial order is not helpful for
# this problem. Stick with cubic order polynomials and RBFs 
order = 0

# domain vertices
vert = np.array([[-1.0,-1.0],
                 [1.0,-1.0],
                 [1.0,1.0],
                 [-1.0,1.0]])
smp = np.array([[0,1],[1,2],[2,3],[3,0]])
# 1 = fixed
# 2 = free
grp = np.array([1,1,2,1])

# fault vertices
vert_f = np.array([[0.5,-0.25],
                   [-0.5,0.75]])
smp_f =  np.array([[0,1]])

# density function
@density_normalizer(vert,smp,N)
def rho(p):
  out = 1.0/(1 + 10*np.linalg.norm(p-np.array([-0.5,0.75]),axis=1)**2)
  out += 1.0/(1 + 10*np.linalg.norm(p-np.array([0.5,-0.25]),axis=1)**2)
  return out

scale = np.max(vert) - np.min(vert)

# fault nodes
nodes_f,smpid_f,is_edge_f = rbf.nodegen.surface(rho,vert_f,smp_f)
simplex_normals = rbf.geometry.simplex_upward_normals(vert_f,smp_f)
norms_f = simplex_normals[smpid_f]
group_f = np.array(is_edge_f,dtype=int)

# cut out any fault nodes outside of the domain
is_inside = contains(nodes_f,vert,smp)
nodes_f = nodes_f[is_inside]
norms_f = norms_f[is_inside]
group_f = group_f[is_inside]

#plt.plot(nodes_f[:,0],nodes_f[:,1],'o')
#plt.quiver(nodes_f[:,0],nodes_f[:,1],norms_f[:,0],norms_f[:,1])
#plt.xlim((-1,1))
#plt.ylim((-1,1))
#plt.show()
# define slip
slip = np.zeros((dim,len(nodes_f[group_f==0])))
knots = np.linspace(-0.25,0.75,8)
basis_no = rbf.bspline.basis_number(knots,2)
slip[1,:] = np.sum(rbf.bspline.bsp1d(nodes_f[group_f==0,1],knots,i,2) for i in range(basis_no))
slip[0,:] = -np.sum(rbf.bspline.bsp1d(nodes_f[group_f==0,1],knots,i,2) for i in range(basis_no))
#plt.plot(nodes_f[group_f==0,1],slip[1,:],'ro')
#plt.plot(nodes_f[group_f==0,1],slip[0,:],'bo')
#plt.show()

# domain nodes
nodes_d,smpid_d = rbf.nodegen.volume(rho,vert,smp,fix_nodes=nodes_f)
simplex_normals = rbf.geometry.simplex_outward_normals(vert,smp)
norms_d = simplex_normals[smpid_d]
norms_d[smpid_d<0] = 0
group_d = np.zeros(nodes_d.shape[0],dtype=int)
group_d[smpid_d==0] = 1
group_d[smpid_d==1] = 1
group_d[smpid_d==2] = 2
group_d[smpid_d==3] = 1

# split fault nodes into hanging wall and foot wall nodes
# THROW AWAY THE END NODES!
nodes_fh = nodes_f[group_f==0] + 1e-10*scale*norms_f[group_f==0]
nodes_ff = nodes_f[group_f==0] - 1e-10*scale*norms_f[group_f==0]
norms_fh = norms_f[group_f==0]
norms_ff = norms_f[group_f==0]

nodes,ix = rbf.nodegen.merge_nodes(interior=nodes_d[group_d==0],
                                   fixed=nodes_d[group_d==1],
                                   free=nodes_d[group_d==2],
                                   free_ghost=nodes_d[group_d==2],
                                   fault_hanging=nodes_fh[...],
                                   fault_hanging_ghost=nodes_fh[...],
                                   fault_foot=nodes_ff[...],
                                   fault_foot_ghost=nodes_ff[...])

norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   fixed=norms_d[group_d==1],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_d[group_d==2],
                                   fault_hanging=norms_fh[...],
                                   fault_hanging_ghost=norms_fh[...],
                                   fault_foot=norms_ff[...],
                                   fault_foot_ghost=norms_ff[...])

# find the node stencils
s,dx = rbf.stencil.nearest(nodes,nodes,Ns,vert=vert_f,smp=smp_f)

# Create a list of distances to the next closest node. This is used to
# determine how far ghost nodes need to be shifted
dx_next_closest = dx[:,[2]]

# shift the ghost nodes outside        
nodes[ix['free_ghost']] += dx_next_closest[ix['free_ghost']]*norms[ix['free_ghost']]

# move hanging wall ghost nodes towards the foot wall and vice versa
nodes[ix['fault_hanging_ghost']] -= dx_next_closest[ix['fault_hanging_ghost']]*norms[ix['fault_hanging_ghost']]
nodes[ix['fault_foot_ghost']] += dx_next_closest[ix['fault_foot_ghost']]*norms[ix['fault_foot_ghost']]


plt.plot(nodes[:,0],nodes[:,1],'ko')
plt.show()
# view nodes
#for i in ix['fault_foot']:
#  plt.plot(nodes[:,0],nodes[:,1],'ko')
#  plt.plot(nodes[s[i],0],nodes[s[i],1],'bo')
#  plt.plot(nodes[i,0],nodes[i,1],'ro')
#  plt.show()

N = len(nodes)

def form_Gij(indices):
  di,mi = indices
  G = scipy.sparse.lil_matrix((N,N),dtype=np.float64)
  for i in ix['interior']:
    w = rbf_weight(nodes[i],
                   nodes[s[i]],
                   evaluate_coeffs_and_diffs(DiffOps[di][mi],i),
                   order=order,
                   basis=basis)

    G[i,s[i]] = w

  for i in ix['free']:
    w = rbf_weight(nodes[i],
                   nodes[s[i]],
                   evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                   order=order,
                   basis=basis)

    G[i,s[i]] = w

  for i in ix['fixed']:
    w = rbf_weight(nodes[i],
                   nodes[s[i]],
                   evaluate_coeffs_and_diffs(FixBCOps[di][mi],i),
                   order=order,
                   basis=basis)

    G[i,s[i]] = w

  # use the hanging wall node indices to impose equal shear tractions 
  # on either side of the fault. use the foot wall to impose the 
  # slip boundary condition
  for itr,hanging_i in enumerate(ix['fault_hanging']):
    foot_i = ix['fault_foot'][itr]
    # weights used to compute traction force on hanging wall
    hanging_w = rbf_weight(nodes[hanging_i],
                   nodes[s[hanging_i]],
                   evaluate_coeffs_and_diffs(FreeBCOps[di][mi],hanging_i),
                   order=order,
                   basis=basis)

    # weights used to compute traction force on foot wall
    foot_w = rbf_weight(nodes[foot_i],
                   nodes[s[foot_i]],
                   evaluate_coeffs_and_diffs(FreeBCOps[di][mi],foot_i),
                   order=order,
                   basis=basis)

    # have the hanging wall fault index difference the shear tractions
    G[hanging_i,s[hanging_i]] = hanging_w
    G[hanging_i,s[foot_i]] = -foot_w

    # use the foot wall fault index to difference the displacements
    hanging_w = rbf_weight(nodes[hanging_i],
                   nodes[s[hanging_i]],
                   evaluate_coeffs_and_diffs(FixBCOps[di][mi],hanging_i),
                   order=order,
                   basis=basis)

    # weights used to compute traction force on foot wall
    foot_w = rbf_weight(nodes[foot_i],
                   nodes[s[foot_i]],
                   evaluate_coeffs_and_diffs(FixBCOps[di][mi],foot_i),
                   order=order,
                   basis=basis)

    G[foot_i,s[hanging_i]] = hanging_w
    G[foot_i,s[foot_i]] = -foot_w
    
  # use the ghost node rows to enforce the free boundary conditions
  # at the boundary nodes
  for itr,i in enumerate(ix['free_ghost']):
    j = ix['free'][itr]
    w = rbf_weight(nodes[j],
                   nodes[s[j]],
                   evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                   order=order,
                   basis=basis)
    G[i,s[j]] = w

  for itr,i in enumerate(ix['fault_hanging_ghost']):
    j = ix['fault_hanging'][itr]
    w = rbf_weight(nodes[j],
                   nodes[s[j]],
                   evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                   order=order,
                   basis=basis)

    G[i,s[j]] = w

  for itr,i in enumerate(ix['fault_foot_ghost']):
    j = ix['fault_foot'][itr]
    w = rbf_weight(nodes[j],
                   nodes[s[j]],
                   evaluate_coeffs_and_diffs(DiffOps[di][mi],j),
                   order=order,
                   basis=basis)

    G[i,s[j]] = w

  return G

# Ensure that each process only uses one thread. The program will
# break otherwise
mkl.set_num_threads(1)
# initiate pool
P = mp.Pool()
# form stiffness matrix in parallel
G_flat = map(form_Gij,[(di,mi) for di in range(dim) for mi in range(dim)])
# reset so that the main process threads use the maximum number of threads
mkl.set_num_threads(mkl.get_max_threads())

G = [[G_flat.pop(0) for i in range(dim)] for j in range(dim)]
data = [np.zeros(N) for i in range(dim)]    
data = np.array(data)
data[:,ix['fault_foot']] = slip
init_guess = np.zeros((dim,N))
init_guess[:,ix['fault_foot']] = -0.5*slip
init_guess[:,ix['fault_hanging']] = 0.5*slip

data = np.concatenate(data)
init_guess = np.concatenate(init_guess)

G = [scipy.sparse.hstack(G[i]) for i in range(dim)]
G = scipy.sparse.vstack(G)

#plt.imshow(np.log10(np.abs(G.toarray())))
#plt.show()

idx_noghost = ix['interior'] + ix['free'] + ix['fixed'] + ix['fault_hanging'] + ix['fault_foot']
out = solver(G,data,init_guess)
out = np.reshape(out,(dim,N))
#out[:,ix['fault_foot']] = out[:,ix['fault_foot']] - slip
#out[:,ix['fault_hanging']] = out[:,ix['fault_foot']] + 2*slip
fig,ax = plt.subplots()
cs = ax.tripcolor(nodes[idx_noghost,0],
                  nodes[idx_noghost,1],
                  np.linalg.norm(out[:,idx_noghost],axis=0),cmap=myplot.cm.slip2)
#plt.colorbar(cs)
plt.quiver(nodes[idx_noghost[::1],0],nodes[idx_noghost[::1],1],
           out[0,idx_noghost[::1]],out[1,idx_noghost[::1]],color='k',scale=40.0)

ax.set_xlim(-0.7,0.7)
ax.set_ylim(-0.5,1.2)
ax.plot([-1.0,1.0],[1.0,1.0],'k-',lw=2)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
logging.basicConfig(level=logging.INFO)

fig2,ax2 = plt.subplots()
ax2.set_xlim(-0.7,0.7)
ax2.set_ylim(-0.5,1.2)
ax2.plot([-1.0,1.0],[1.0,1.0],'k-',lw=2)
ax2.plot([-0.5,0.5],[0.75,-0.25],'r-',lw=2)
ax2.plot(nodes[idx_noghost,0],nodes[idx_noghost,1],'ko')

ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)

#fig,ax = plt.subplots()
#ax.plot(nodes[ix['free'],0],out[1,ix['free']],'o')
#ax.plot(nodes[ix['free'],0],out[0,ix['free']],'o')
plt.show()


