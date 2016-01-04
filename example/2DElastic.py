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
import rbf.bspline
from myplot.cm import slip2
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import logging
from modest import summary
import sympy as sp
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

  # form jacobi preconditioner
  diag = np.array(G[range(G.shape[0]),range(G.shape[0])])[0]
  if np.any(diag == 0.0):
    raise ValueError(
      'matrix cannot be sorted into a diagonal dominant matrix')

  M = scipy.sparse.diags(1.0/diag,0)
  out,status = scipy.sparse.linalg.lgmres(G,d,M=M,maxiter=5000)
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
           L.diff(x[0]):0.0,
           L.diff(x[1]):lamb_diffy,
           M.diff(x[0]):0.0,
           M.diff(x[1]):lamb_diffy,
           n[0]:lambda i:norms[i,0],
           n[1]:lambda i:norms[i,1]}

DiffOps = [[coeffs_and_diffs(PDEs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FreeBCOps = [[coeffs_and_diffs(FreeBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]
FixBCOps = [[coeffs_and_diffs(FixBCs[i],u[j],x,mapping=sym2num) for j in range(dim)] for i in range(dim)]

cond=10
N = 10000
Ns = 20
Np = 1

# domain vertices

surf_vert_x = np.linspace(-10,10,100)
surf_vert_y = 1.0/(1 + 1.0*(surf_vert_x - 2)**2)

surf_vert = np.array([surf_vert_x,surf_vert_y+10]).T
surf_smp = np.array([np.arange(99),np.arange(1,100)]).T
bot_vert = np.array([[-10.0,-10.0],
                     [10.0,-10.0]])
vert = np.vstack((bot_vert,surf_vert))
smp = np.concatenate(([[0,1],[0,2],[1,101]],surf_smp+2))
smp = np.array(smp,dtype=int)
print(len(smp))
for s in smp:
  plt.plot(vert[s,0],vert[s,1],'bo-')
plt.show()  
# 1 = fixed
# 2 = free
grp = 2*np.ones(102)
grp[[0,1,2]] = 1
grp = np.array(grp,dtype=int)

# fault vertices
vert_f = np.array([[-1.0,9.0],
                   [0.5,10.35]])
smp_f =  np.array([[0,1]])


# density function
@normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  out = np.zeros(p.shape[0])
  for v in vert_f:
    out +=  1.0/(1 + 2.*np.linalg.norm(p - v,axis=-1)**2)

  out +=  1.0/(1 + 2.*np.linalg.norm(p - np.mean(vert_f,axis=0),axis=-1)**2)
  return out

scale = np.max(vert) - np.min(vert)

# fault nodes
nodes_f,norms_f,group_f = rbf.nodegen.surface(rho,vert_f,smp_f)

# cut out any fault nodes outside of the domain
is_inside = boundary_contains(nodes_f,vert,smp)
nodes_f = nodes_f[is_inside]
norms_f = norms_f[is_inside]
group_f = group_f[is_inside]

# define slip
slip = np.zeros((dim,len(nodes_f)))
slip[0,:] = -rbf.bspline.bsp1d(nodes_f[:,1],9.8+np.array([0.0,0.33,0.66,1.0]),0,2)
slip[1,:] = -rbf.bspline.bsp1d(nodes_f[:,1],9.8+np.array([0.0,0.33,0.66,1.0]),0,2)
#slip[1,:] = rbf.bspline.bsp1d(nodes_f[:,1],-0.5+np.array([0.0,0.33,0.66,1.0]),0,2)
#slip[0,:] = 1.0
#slip[1,:] = 1.0

# ensures that there is no slip on end nodes
slip[0,group_f==1]=0
slip[1,group_f==1]=0

plt.plot(nodes_f[:,1],slip[1,:],'ro')
plt.plot(nodes_f[:,1],slip[0,:],'bo')
plt.show()

# domain nodes
nodes_d,norms_d,group_d = rbf.nodegen.volume(rho,vert,smp,groups=grp,
                                             fix_nodes=nodes_f)

# ghost nodes for free surfaces
nodes_g = nodes_d[group_d==2]
norms_g = norms_d[group_d==2]

# split fault nodes into hanging wall and foot wall nodes
nodes_fh = nodes_f + 1e-10*scale*norms_f
nodes_ff = nodes_f - 1e-10*scale*norms_f
norms_fh = norms_f
norms_ff = norms_f

nodes,ix = rbf.nodegen.merge_nodes(interior=nodes_d[group_d==0],
                                   fixed=nodes_d[group_d==1],
                                   free=nodes_d[group_d==2],
                                   free_ghost=nodes_g,
                                   fault_hanging=nodes_fh,
                                   fault_foot=nodes_ff)

norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   fixed=norms_d[group_d==1],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_g,
                                   fault_hanging=norms_fh,
                                   fault_foot=norms_ff)

s,dx = rbf.stencil.nearest(nodes,nodes,Ns,vert_f,smp_f)

# replace fault stencils so that they do not include other fault nodes                                             
idx_nofault = ix['interior'] + ix['fixed'] + ix['free'] + ix['free_ghost']
idx_noghost = ix['interior'] + ix['fixed'] + ix['free'] + ix['fault_hanging'] + ix['fault_foot']

sfh,dxfh = rbf.stencil.nearest(nodes[ix['fault_hanging']],nodes[idx_nofault],Ns-1,vert_f,smp_f)
sfh = np.array(idx_nofault)[sfh]

sff,dxff = rbf.stencil.nearest(nodes[ix['fault_foot']],nodes[idx_nofault],Ns-1,vert_f,smp_f)
sff = np.array(idx_nofault)[sff]

sfh = np.hstack((np.array(ix['fault_hanging'])[:,None],sfh))
sff = np.hstack((np.array(ix['fault_foot'])[:,None],sff))
s[ix['fault_hanging']] = sfh
s[ix['fault_foot']] = sff


# find the distance to nearest node ignoring duplicate vertices
dx[dx==0.0] = np.inf
min_dx = np.min(dx,axis=-1)

# shift the ghost nodes outside
nodes[ix['free_ghost']] = nodes[ix['free_ghost']] + min_dx[ix['free_ghost'],None]*norms[ix['free_ghost']]

# view nodes
plt.plot(nodes[:,0],nodes[:,1],'o')
plt.show()

# find the optimal shape factor for each stencil
eps = rbf.weights.shape_factor(nodes,s,basis,cond=cond,samples=200)

N = len(nodes)
modest.tic('forming G')

G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float64) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N) for i in range(dim)]
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

modest.toc('forming G')

out = solver(G,data)
out = np.reshape(out,(dim,N))
out[:,ix['fault_foot']] = out[:,ix['fault_foot']] - slip
out[:,ix['fault_hanging']] = out[:,ix['fault_foot']] + 2*slip
fig,ax = plt.subplots()
cs = ax.tripcolor(nodes[idx_noghost,0],
                  nodes[idx_noghost,1],
                  np.linalg.norm(out[:,idx_noghost],axis=0),cmap=slip2,vmin=0,vmax=1.0)
plt.quiver(nodes[idx_noghost[::2],0],nodes[idx_noghost[::2],1],
           out[0,idx_noghost[::2]],out[1,idx_noghost[::2]])

logging.basicConfig(level=logging.INFO)
summary()

plt.show()


