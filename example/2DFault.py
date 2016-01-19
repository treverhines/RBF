#!/usr/bin/env python
import numpy as np
import rbf.nodegen
from rbf.basis import phs3 as basis
from rbf.normalize import normalizer
from rbf.geometry import complex_contains
import modest
from rbf.weights import rbf_weight
import rbf.stencil
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs_and_diffs
import numpy as np
import rbf.bspline
from myplot.cm import slip2
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import logging
from modest import summary
import sympy as sp
import multiprocessing as mp
import mkl
logging.basicConfig(level=logging.INFO)



@modest.funtime
def solver(G,d):
  if not scipy.sparse.isspmatrix_csc(G):
    G = scipy.sparse.csc_matrix(G)

  # sort G and d using reverse cuthill mckee algorithm
  perm = scipy.sparse.csgraph.reverse_cuthill_mckee(G)  
  rev_perm = np.argsort(perm)
  G = G[perm,:]
  G = G[:,perm]
  d = d[perm]

  # it is not clear whether sorting the matrix helps much
  out = scipy.sparse.linalg.spsolve(G,d)

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
N = 2000
Ns = 50
# It seems like using the maximum polynomial order is not helpful for
# this problem. Stick with cubic order polynomials and RBFs 
order = 4

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
vert_f = np.array([[0.0,-0.5],
                   [0.0001,0.5]])
smp_f =  np.array([[0,1]])

# density function
@normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  out = 1.0/(1 + 50*np.linalg.norm(p-np.array([0.0,0.5]),axis=1)**2)
  out += 1.0/(1 + 50*np.linalg.norm(p-np.array([0.0,-0.5]),axis=1)**2)
  return out

scale = np.max(vert) - np.min(vert)

# fault nodes
nodes_f,norms_f,group_f = rbf.nodegen.surface(rho,vert_f,smp_f)

# cut out any fault nodes outside of the domain
is_inside = complex_contains(nodes_f,vert,smp)
nodes_f = nodes_f[is_inside]
norms_f = norms_f[is_inside]
group_f = group_f[is_inside]

# define slip
slip = np.zeros((dim,len(nodes_f[group_f==0])))
knots = np.linspace(-0.5,0.5,6)
basis_no = rbf.bspline.basis_number(knots,2)
slip[1,:] = np.sum(rbf.bspline.bsp1d(nodes_f[group_f==0,1],knots,i,2) for i in range(basis_no))

plt.plot(nodes_f[group_f==0,1],slip[1,:],'ro')
plt.plot(nodes_f[group_f==0,1],slip[0,:],'bo')
plt.show()

# domain nodes
nodes_d,norms_d,group_d = rbf.nodegen.volume(rho,vert,smp,groups=grp,
                                             fix_nodes=nodes_f,itr=20,delta=0.1,n=10)

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
modest.tic('forming G')

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

  # treat fault nodes as free nodes and the later reorganize the G
  # matrix so that the fault nodes are forces to be equal
  for i in ix['fault_hanging']+ix['fault_foot']:
    w = rbf_weight(nodes[i],
                   nodes[s[i]],
                   evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                   order=order,
                   basis=basis)

    G[i,s[i]] = w

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
G_flat = P.map(form_Gij,[(di,mi) for di in range(dim) for mi in range(dim)])
# reset so that the main process threads use the maximum number of threads
mkl.set_num_threads(mkl.get_max_threads())

G = [[G_flat.pop(0) for i in range(dim)] for j in range(dim)]
data = [np.zeros(N) for i in range(dim)]    


# adjust G and d to allow for split nodes
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

idx_noghost = ix['interior'] + ix['free'] + ix['fixed'] + ix['fault_hanging'] + ix['fault_foot']
modest.toc('forming G')
out = solver(G,data)
out = np.reshape(out,(dim,N))
out[:,ix['fault_foot']] = out[:,ix['fault_foot']] - slip
out[:,ix['fault_hanging']] = out[:,ix['fault_foot']] + 2*slip
fig,ax = plt.subplots()
cs = ax.tripcolor(nodes[idx_noghost,0],
                  nodes[idx_noghost,1],
                  np.linalg.norm(out[:,idx_noghost],axis=0),cmap=slip2)
plt.colorbar(cs)
plt.quiver(nodes[idx_noghost[::1],0],nodes[idx_noghost[::1],1],
           out[0,idx_noghost[::1]],out[1,idx_noghost[::1]],color='k',scale=20.0)



logging.basicConfig(level=logging.INFO)
summary()

plt.show()


