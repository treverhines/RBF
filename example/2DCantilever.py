#!/usr/bin/env python
import numpy as np
import rbf.nodegen
from rbf.basis import phs5 as basis
from rbf.integrate import density_normalizer
from rbf.geometry import contains
from rbf.weights import rbf_weight
import rbf.stencil
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs_and_diffs
import numpy as np
import rbf.bspline
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import logging
import sympy as sp
import matplotlib.cm
logging.basicConfig(level=logging.INFO)


# end-user parameters
######################################################################
# number of nodes to use 
N = 2000
# number of nodes per stencil
Ns = 20
# order of the added polynomial terms
order = 2

# define the domain vertices
vert = np.array([[0.0,0.0],
                 [10.0,0.0],
                 [10.0,5.0],
                 [00.0,5.0]])

# define the domain simplices
smp = np.array([[0,1],[1,2],[2,3],[3,0]])

# specify the boundary conditions for each simplex. 1 makes it fixed
# at 0 and 2 makes it free
bc = np.array([2,2,2,1])

# lame parameters 
lamb = 1.0
mu = 1.0

# grativational force
grav = -1.0

# node density function. rho takes a (N,2) array and returns the node
# density at that point. the decorator is normalizes the function so
# that it integrates to N
@density_normalizer(vert,smp,N)
def rho(p):
  return 1.0 + 0.2*p[:,0]

######################################################################

def solver(G,d):
  if not scipy.sparse.isspmatrix_csc(G):
    G = scipy.sparse.csc_matrix(G)

  # sort G and d using reverse cuthill mckee algorithm
  perm = scipy.sparse.csgraph.reverse_cuthill_mckee(G)  
  rev_perm = np.argsort(perm)
  G = G[perm,:]
  G = G[:,perm]
  d = d[perm]

  out = scipy.sparse.linalg.spsolve(G,d)

  # return to original sorting
  d = d[rev_perm]
  out = out[rev_perm]
  G = G[rev_perm,:]
  G = G[:,rev_perm]
  return out   


# formulate the PDE using sympy
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

# define a mapping from symbolic expressions to numerical functions or
# scalars. In this case, we are assuming the Lame parameters, L and M,
# are homogeneous.
sym2num = {L:lamb,
           M:mu,
           sp.Integer(1):1.0,
           sp.Integer(2):2.0,
           L.diff(x[0]):0.0,
           L.diff(x[1]):0.0,
           M.diff(x[0]):0.0,
           M.diff(x[1]):0.0,
           n[0]:lambda i:norms[i,0],
           n[1]:lambda i:norms[i,1]}

DiffOps = [[coeffs_and_diffs(PDEs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FreeBCOps = [[coeffs_and_diffs(FreeBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FixBCOps = [[coeffs_and_diffs(FixBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]

scale = np.max(vert) - np.min(vert)

# domain nodes
nodes,smpid = rbf.nodegen.volume(rho,vert,smp)

# find normal vectors for each simplex
smp_norms = rbf.geometry.simplex_normals(vert,smp)

# surface normal vectors for each node. vectors for interior nodes are zero
norms = smp_norms[smpid]
norms[smpid==-1] = 0

# boundary condition for each node
node_bc = bc[smpid]
node_bc[smpid==-1] = 0

nodes,ix = rbf.nodegen.merge_nodes(interior=nodes[node_bc==0],
                                   fixed=nodes[node_bc==1],
                                   free=nodes[node_bc==2],
                                   free_ghost=nodes[node_bc==2])

norms,ix = rbf.nodegen.merge_nodes(interior=norms[node_bc==0],
                                   fixed=norms[node_bc==1],
                                   free=norms[node_bc==2],
                                   free_ghost=norms[node_bc==2])

# find the nearest neighbors for the ghost nodes
s,dx = rbf.stencil.nearest(nodes[ix['free_ghost']],nodes,3)

# The closest nodes are going the be the free nodes, which currently 
# are on top of the ghost nodes. find the distance to the next closest 
# node
dx = dx[:,[2]]

# shift the ghost nodes outside        
nodes[ix['free_ghost']] += dx*norms[ix['free_ghost']]
plt.plot(nodes[:,0],nodes[:,1],'o')
plt.show()

# find the stencils and distances now that the ghost nodes have been
# moved
s,dx = rbf.stencil.nearest(nodes,nodes,Ns)

N = len(nodes)

G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float32) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N,dtype=np.float32) for i in range(dim)]
# This can be parallelized!!!!
for di in range(dim):
  for mi in range(dim):
    # apply the PDE to interior nodes and free nodes
    for i in ix['interior']+ix['free']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],i),
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

    # use the ghost node rows to enforce the free boundary conditions
    # at the boundary nodes
    for itr,i in enumerate(ix['free_ghost']):
      j = ix['free'][itr]
      w = rbf_weight(nodes[j],
                     nodes[s[j]],
                     evaluate_coeffs_and_diffs(FreeBCOps[di][mi],j),
                     order=order,
                     basis=basis)
      G[di][mi][i,s[j]] = w
    

G = [scipy.sparse.hstack(G[i]) for i in range(dim)]
G = scipy.sparse.vstack(G)

# add gravitational force to indices where the PDE was enforced
data[1][ix['interior']] -= grav
data[1][ix['free']] -= grav

data = np.concatenate(data)

idx_noghost = ix['free'] + ix['fixed'] + ix['interior']
out = solver(G,data)

# Displacement solution
out = np.reshape(out,(dim,N))

# plot the results
fig,ax = plt.subplots()
cs = ax.tripcolor(nodes[idx_noghost,0],
                  nodes[idx_noghost,1],
                  np.linalg.norm(out[:,idx_noghost],axis=0),cmap=matplotlib.cm.cubehelix)
plt.colorbar(cs)
plt.quiver(nodes[idx_noghost[::1],0],nodes[idx_noghost[::1],1],
           out[0,idx_noghost[::1]],out[1,idx_noghost[::1]],color='k')

plt.show()


