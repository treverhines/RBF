#!/usr/bin/env python
import numpy as np
import rbf.nodegen
from rbf.basis import mq as basis
from rbf.normalize import normalizer
from rbf.geometry import boundary_contains
import modest
from rbf.weights import rbf_weight
import rbf.stencil
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs_and_diffs
import numpy as np
import mayavi.mlab
from myplot.cm import slip2
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import logging
from modest import summary
import random
import sympy as sp
logging.basicConfig(level=logging.INFO)


@modest.funtime
def jacobi_preconditioned_solve(G,d):
  if not scipy.sparse.isspmatrix_csc(G):
    G = scipy.sparse.csc_matrix(G)

  # sort G and d using reverse cuthill mckee algorithm
  perm = scipy.sparse.csgraph.reverse_cuthill_mckee(G)
  rev_perm = np.argsort(perm)
  G = G[perm,:]
  G = G[:,perm]
  d = d[perm]

  # form jacobi preconditioner       
  diag = np.array(G[range(G.shape[0]),range(G.shape[0])])[0]
  if np.any(diag == 0.0):
    raise ValueError(
      'matrix cannot be sorted into a diagonal dominant matrix')

  M = scipy.sparse.diags(1.0/diag,0)
  #out,status = scipy.sparse.linalg.lgmres(G,d,M=M)
  out = scipy.sparse.linalg.spsolve(G,d)
  #if status != 0:
  #  raise ValueError(
  #    'lgmres exited with status %s' % status)

  # return to original sorting
  d = d[rev_perm]
  out = out[rev_perm]
  G = G[rev_perm,:]
  G = G[:,rev_perm]
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
sp.pprint(PDEs)

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

cond=3
shape_factor=0.8
N = 10000

# values of Ns=8 are needed when slip has sharp discontinuities. 
# Much higher values can be used for even C1 continuous slip
Ns = 15
Np = 1

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
                [4,6,7],
                [2,3,7],
                [2,6,7]])

grp = np.ones(len(smp))
# make top a free surface
grp[2] = 2
grp[3] = 2

vert_f = np.array([[0.5,0.25,0.5],
                   [0.5,0.25,1.0],
                   [0.5,0.75,1.0],
                   [0.5,0.75,0.5]])
smp_f =  np.array([[0,1,2],
                   [0,2,3]])


@normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  out = np.zeros(p.shape[0])
  out += 1.0/(1.0 + 100*np.linalg.norm(p-np.array([1.0,1.0,1.0]),axis=1)**2)

  return out

scale = np.max(vert) - np.min(vert)

# fault nodes
nodes_f,norms_f,group_f = rbf.nodegen.surface(rho,vert,smp)
mayavi.mlab.points3d(nodes_f[:,0],nodes_f[:,1],nodes_f[:,2],scale_factor=0.005)
mayavi.mlab.show()

# cut out any fault nodes outside of the domain
is_inside = boundary_contains(nodes_f,vert,smp)
nodes_f = nodes_f[is_inside]
norms_f = norms_f[is_inside]
group_f = group_f[is_inside]

slip = np.zeros((dim,len(nodes_f)))
knots_z = np.linspace(0.5,1.5,4)
knots_y = np.linspace(0.25,0.75,4)
import rbf.bspline
basis_no = rbf.bspline.basis_number(knots_z,2)
slip[1,:] = rbf.bspline.bspnd(nodes_f[:,[1,2]],(knots_y,knots_z),(0,0),(2,2))

plt.plot(slip[1,:])
plt.show()
mayavi.mlab.points3d(nodes_f[:,0],nodes_f[:,1],nodes_f[:,2],slip[1,:])
mayavi.mlab.show()
# domain nodes
nodes_d,norms_d,group_d = rbf.nodegen.volume(rho,vert,smp,groups=grp,
                                             fix_nodes=nodes_f)

s,dx = rbf.stencil.nearest(nodes_d,nodes_d,2)
plt.hist(dx[:,1],100)
plt.show()
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
                                   fault_foot=nodes_ff)



norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_d[group_d==2],
                                   fixed=norms_d[group_d==1],
                                   fault_hanging=norms_fh,
                                   fault_foot=norms_ff)


mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,opacity=0.5,color=(1,0,0))
mayavi.mlab.triangular_mesh(vert_f[:,0],vert_f[:,1],vert_f[:,2],smp_f,opacity=0.5,color=(1,0,0))
for k,v in ix.iteritems():
  mayavi.mlab.points3d(nodes[v,0],nodes[v,1],nodes[v,2],scale_factor=0.01)

print(k)
mayavi.mlab.show()


# find the nearest neighbors for the ghost nodes          
s,dx = rbf.stencil.nearest(nodes[ix['free_ghost']],nodes,3,vert=vert_f,smp=smp_f)

# The closest nodes are going the be the free nodes, which currently      
# are on top of the ghost nodes. find the distance to the next closest    
# node                                                              
dx = dx[:,[2]]

# shift the ghost nodes outside           
nodes[ix['free_ghost']] += dx*norms[ix['free_ghost']]

# find the stencils and distances now that the ghost nodes have been    
# moved                             
s,dx = rbf.stencil.nearest(nodes,nodes,Ns,vert=vert_f,smp=smp_f)

# change stencils so that fault nodes, where traction forces are  
# estimated, do not include other fault nodes. This greatly improves     
# accuracy and reduces the odds of instability when slip has sharp   
# discontinuities.  Adding ghost nodes to the fault does not improve    
# the solution and may make it worse              
fault_indices = np.array(ix['fault_hanging']+ix['fault_foot'])
fault_stencil = rbf.stencil.nearest(nodes[fault_indices],nodes,Ns-1,
                                    vert=vert_f,smp=smp_f,
                                    excluding=fault_indices)[0]
# make sure the stencil includes itself                
fault_stencil = np.hstack((fault_indices[:,None],fault_stencil))
s[fault_indices] = fault_stencil


# find the optimal shape factor for each stencil
eps = rbf.weights.shape_factor(nodes,s,basis,alpha=shape_factor,cond=cond,samples=200)

N = len(nodes)

G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float64) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N) for mi in range(dim)]
modest.tic('forming G')
# This can be parallelized!!!!
for di in range(dim):
  for mi in range(dim):
    for i in ix['interior']+ix['free']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(DiffOps[di][mi],i),
                     eps=eps[i],
                     Np=Np,
                     basis=basis,
                     cond=cond)
      G[di][mi][i,s[i]] = w

    for i in ix['fixed']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(FixBCOps[di][mi],i),
                     eps=eps[i],
                     Np=Np,
                     basis=basis,
                     cond=cond)
      G[di][mi][i,s[i]] = w

    for i in ix['fault_hanging']+ix['fault_foot']:
      w = rbf_weight(nodes[i],
                     nodes[s[i]],
                     evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                     eps=eps[i],
                     Np=Np,
                     basis=basis,
                     cond=cond)
      G[di][mi][i,s[i]] = w

    for itr,i in enumerate(ix['free_ghost']):
      j = ix['free'][itr]
      w = rbf_weight(nodes[j],
                     nodes[s[j]],
                     evaluate_coeffs_and_diffs(FreeBCOps[di][mi],j),
                     eps=eps[j],
                     Np=Np,
                     basis=basis,
                     cond=cond)
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

modest.toc('forming G')


G = G.tocsc()
out = jacobi_preconditioned_solve(G,data)
out = np.reshape(out,(dim,N))
out = out.T

import matplotlib.pyplot as plt
plt.quiver(nodes[ix['free'],0],nodes[ix['free'],1],out[ix['free'],0],out[ix['free'],1])
plt.show()
idx = ix['free'] + ix['interior'] + ix['fixed']
mayavi.mlab.quiver3d(nodes[idx,0],nodes[idx,1],nodes[idx,2],out[idx,0],out[idx,1],out[idx,2],mode='arrow',color=(0,1,0),scale_factor=0.1)


mayavi.mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],smp,opacity=0.2,color=(1,1,1))
mayavi.mlab.triangular_mesh(vert_f[:,0],vert_f[:,1],vert_f[:,2],smp_f,opacity=0.2,color=(1,0,0))

mayavi.mlab.show()
logging.basicConfig(level=logging.INFO)
summary()




