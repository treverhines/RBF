#!/usr/bin/env python
import numpy as np
import rbf.nodegen
from rbf.basis import phs5 as basis
from rbf.integrate import density_normalizer
from rbf.geometry import contains
from rbf.fd import diff_weights
import rbf.stencil
import myplot.cm
from rbf.halton import Halton
from rbf.formulation import coeffs_and_diffs
from rbf.formulation import evaluate_coeffs
import numpy as np
import rbf.bspline
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.sparse
import scipy.sparse.linalg
import logging
from modest import summary
import sympy as sp
logging.basicConfig(level=logging.INFO)
import sys
import modest
import petsc4py 
petsc4py.init(sys.argv)
from petsc4py import PETSc

def solver(G,d):
  N = G.shape[0]
  #G += 10.1*scipy.sparse.eye(N)
  G = G.tocsr()
  G = G.astype(np.float64)
  d = d.astype(np.float64)
  A = PETSc.Mat().createAIJ(size=G.shape,csr=(G.indptr,G.indices,G.data))
  d = PETSc.Vec().createWithArray(d)
  #soln = scipy.sparse.linalg.spsolve(G,d)
  soln = np.zeros(G.shape[1]) + 0.0
  soln = PETSc.Vec().createWithArray(soln)

  #plt.plot(d)
  #plt.show()
  ksp = PETSc.KSP()
  ksp.create()
  ksp.rtol = 1e-10
  ksp.atol = 1e-10
  ksp.max_it = 10000
  ksp.setType('gmres')
  #ksp.setRestart(100)
  ksp.setInitialGuessNonzero(True)
  #ksp.setInitialGuessKnoll(True)
  ksp.setOperators(A)
  ksp.setFromOptions()
  pc = ksp.getPC()
  #pc.setType('none')
  pc.setUp()
  ksp.solve(d,soln)
  ksp.view()
  return soln.getArray()

def _solver(G,d):
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
  out,status = scipy.sparse.linalg.lgmres(G,d,M=M,maxiter=1)
  if status != 0:
    print('lgmres exited with status %s, trying direct solve' % status)
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

# The number of nodes needed will depend entirely on how sharply slip varies
N = 2000
Ns = 50
order = 4#'max'

# domain vertices
surf_vert_x = np.linspace(-5,5,200)
surf_vert_y = 1.0/(1 + 1.0*(surf_vert_x)**2)
#surf_vert_y = 2.0*np.sin(4*np.pi*surf_vert_x/10.0)

surf_vert = np.array([surf_vert_x,surf_vert_y+10]).T
surf_smp = np.array([np.arange(199),np.arange(1,200)]).T
bot_vert = np.array([[-5.0,-5.0],
                     [5.0,-5.0]])
#### for plotting purposes onlys
top_vert = np.array([[5.0,11.5],[-5.0,11.5]])
####
vert = np.vstack((bot_vert,surf_vert))
anti_vert = np.vstack((top_vert,surf_vert))
smp = np.concatenate(([[0,1],[0,2],[1,201]],surf_smp+2))
smp = np.array(smp,dtype=int)
grp = 2*np.ones(len(smp))
grp[[0,1,2]] = 1

# 1 = fixed
# 2 = free

# density function
@density_normalizer(vert,smp,N)
def rho(p):
  out = 1.0/(1 + 0.1*np.linalg.norm(p - np.array([0.0,11.0]),axis=1)**2)
  out += 1.0/(1 + 0.1*np.linalg.norm(p - np.array([2.0,10.0]),axis=1)**2)
  out += 1.0/(1 + 0.1*np.linalg.norm(p - np.array([-2.0,10.0]),axis=1)**2)
  return out

scale = np.max(vert) - np.min(vert)

# domain nodes
smp = rbf.geometry.oriented_simplices(vert,smp)
simplex_normals = rbf.geometry.simplex_normals(vert,smp)
#for itr,s in enumerate(smp):
#  plt.plot(vert[s,0],vert[s,1],'o-')
#  plt.quiver(vert[s,0],vert[s,1],simplex_normals[[itr,itr],0],simplex_normals[[itr,itr],1])

#plt.show()
#vert += 0.001*np.random.random(vert.shape)
nodes_d,sidx = rbf.nodegen.volume(rho,vert,smp)
norms_d = simplex_normals[sidx]
group_d = grp[sidx]
group_d[sidx==-1] = 0

#plt.plot(nodes_d[group_d==0,0],nodes_d[group_d==0,1],'bo')
#plt.plot(nodes_d[group_d==1,0],nodes_d[group_d==1,1],'ro')
#plt.plot(nodes_d[group_d==2,0],nodes_d[group_d==2,1],'go')
#plt.show()
nodes,ix = rbf.nodegen.merge_nodes(interior=nodes_d[group_d==0],
                                   fixed=nodes_d[group_d==1],
                                   free=nodes_d[group_d==2],
                                   free_ghost=nodes_d[group_d==2])

norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   fixed=norms_d[group_d==1],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_d[group_d==2])

# find the nearest neighbors for the ghost nodes
s,dx = rbf.stencil.nearest(nodes[ix['free_ghost']],nodes,3)

# The closest nodes are going the be the free nodes, which currently 
# are on top of the ghost nodes. find the distance to the next closest 
# node
dx = dx[:,[2]]

# shift the ghost nodes outside        
nodes[ix['free_ghost']] += dx*norms[ix['free_ghost']]
#plt.plot(nodes[:,0],nodes[:,1],'o')
#plt.show()

# find the stencils and distances now that the ghost nodes have been
# moved
s,dx = rbf.stencil.nearest(nodes,nodes,Ns)

N = len(nodes)

G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float32) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N,dtype=np.float32) for i in range(dim)]

modest.tic('building')
# This can be parallelized!!!!
for di in range(dim):
  for mi in range(dim):
    # apply the PDE to interior nodes and free nodes
    for i in ix['interior']+ix['free']:
      coeffs,diffs = DiffOps[di][mi]
      coeffs_eval = evaluate_coeffs(coeffs,i)
      w = diff_weights(nodes[i],
                       nodes[s[i]],
                       diffs=diffs,  
                       coeffs=coeffs_eval,  
                       order=order,
                       basis=basis)
      G[di][mi][i,s[i]] = w

    for i in ix['fixed']:
      coeffs,diffs = FixBCOps[di][mi]
      coeffs_eval = evaluate_coeffs(coeffs,i)
      w = diff_weights(nodes[i],
                       nodes[s[i]],
                       diffs=diffs,  
                       coeffs=coeffs_eval,  
                       order=order,
                       basis=basis)
      G[di][mi][i,s[i]] = w

    # use the ghost node rows to enforce the free boundary conditions
    # at the boundary nodes
    for itr,i in enumerate(ix['free_ghost']):
      j = ix['free'][itr]
      coeffs,diffs = FreeBCOps[di][mi]
      coeffs_eval = evaluate_coeffs(coeffs,j)
      w = diff_weights(nodes[j],
                       nodes[s[j]],
                       diffs=diffs,  
                       coeffs=coeffs_eval,  
                       order=order,
                       basis=basis)
      G[di][mi][i,s[j]] = w
    
print(modest.toc('building'))

G = [scipy.sparse.hstack(G[i]) for i in range(dim)]
G = scipy.sparse.vstack(G)

# add gravitational force
data[1][ix['interior']] += (nodes[ix['interior'],1] > 10.01).astype(np.float32)
data[1][ix['free']] += (nodes[ix['free'],1] > 10.01).astype(np.float32)
data = np.concatenate(data)

idx_noghost = ix['free'] + ix['fixed'] + ix['interior']
out = solver(G,data)
out = np.reshape(out,(dim,N))
fig,ax = plt.subplots()

inside_nodes = nodes[idx_noghost]
H = Halton(2)
outside_nodes = H(100000)-0.5
outside_nodes[:,0] *= 10
outside_nodes[:,1] += 10.5

outside_nodes = outside_nodes[~contains(outside_nodes,vert,smp)]
soln = out[:,idx_noghost]

nodes = inside_nodes

#cs = ax.tripcolor(nodes[idx_noghost,0],
#                  nodes[idx_noghost,1],
#                  np.linalg.norm(out[:,idx_noghost],axis=0),cmap=myplot.cm.viridis)
cs = ax.tripcolor(nodes[:,0],
                  nodes[:,1],
                  np.linalg.norm(soln,axis=0),cmap=myplot.cm.slip2)
#plt.colorbar(cs)
plt.quiver(nodes[::1,0],nodes[::1,1],
           soln[0,::1],soln[1,::1],color='k',scale=40)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)

ax.set_xlim(-4,4)
ax.set_ylim(3.5,11.5)

for s in smp:
  ax.plot(vert[s,0],vert[s,1],'k-',lw=2)

poly = Polygon(anti_vert,closed=False,color='w')
ax.add_artist(poly)

fig2,ax2 = plt.subplots()

plt.plot(nodes[:,0],nodes[:,1],'ko')
#ax2.set_axis_off()
ax2.set_xlim(-4,4)
ax2.set_ylim(3.5,11.5)

for s in smp:
  ax2.plot(vert[s,0],vert[s,1],'k-',lw=2)

poly = Polygon(anti_vert,closed=False,color='w')
ax2.add_artist(poly)
#ax2.set_frame_on(False)
ax2.axes.get_yaxis().set_visible(False)
ax2.axes.get_xaxis().set_visible(False)

logging.basicConfig(level=logging.INFO)

plt.show()


