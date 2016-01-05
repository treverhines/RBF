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
from elastostatics import okada
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
# The number of nodes needed will depend entirely on how sharply slip varies
N = 100000

# Ns=7 produces fantastic results in 2D because it is the number of 
# adjacent nodes assuming HCP.  but 7 can be dangerous if there is 
# a really shitty mesh 9 is a safer bet 
Ns = 9
Np = 1

# domain vertices

surf_vert_x = np.linspace(-10,10,100)
surf_vert_y = 1.0/(1 + 1.0*(surf_vert_x - 2)**2)
surf_vert_y = 2.0*np.sin(4*np.pi*surf_vert_x/10.0)

surf_vert = np.array([surf_vert_x,surf_vert_y+12]).T
surf_smp = np.array([np.arange(99),np.arange(1,100)]).T
bot_vert = np.array([[-10.0,-10.0],
                     [10.0,-10.0]])
vert = np.vstack((bot_vert,surf_vert))
smp = np.concatenate(([[0,1],[0,2],[1,101]],surf_smp+2))
smp = np.array(smp,dtype=int)

# 1 = fixed
# 2 = free
grp = 2*np.ones(102)
grp[[0,1,2]] = 1
grp = np.array(grp,dtype=int)

# fault vertices
vert_f = np.array([[0.0,9.0],
                   [0.0001,10.1]])
smp_f =  np.array([[0,1]])


# density function
@normalizer(vert,smp,kind='density',nodes=N)
def rho(p):
  out = np.zeros(p.shape[0])
  #for v in vert_f:
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.0]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.1]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.2]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.3]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.4]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.5]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.6]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.7]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.8]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,9.9]),axis=-1)**2)
  out +=  1.0/(1 + 20.*np.linalg.norm(p - np.array([0.0,10.0]),axis=-1)**2)
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
knots = np.linspace(9.0,10.0,4)
basis_no = rbf.bspline.basis_number(knots,2)
slip[0,:] = -np.sum(rbf.bspline.bsp1d(nodes_f[:,1],knots,i,2) for i in range(basis_no))
#slip[1,:] = -np.sum(rbf.bspline.bsp1d(nodes_f[:,1],knots,i,2) for i in range(basis_no))
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

# split fault nodes into hanging wall and foot wall nodes
nodes_fh = nodes_f + 1e-10*scale*norms_f
nodes_ff = nodes_f - 1e-10*scale*norms_f
norms_fh = np.copy(norms_f)
norms_ff = np.copy(norms_f)

nodes,ix = rbf.nodegen.merge_nodes(interior=nodes_d[group_d==0],
                                   fixed=nodes_d[group_d==1],
                                   free=nodes_d[group_d==2],
                                   free_ghost=nodes_d[group_d==2],
                                   fault_hanging=nodes_fh,
                                   fault_foot=nodes_ff)

norms,ix = rbf.nodegen.merge_nodes(interior=norms_d[group_d==0],
                                   fixed=norms_d[group_d==1],
                                   free=norms_d[group_d==2],
                                   free_ghost=norms_d[group_d==2],
                                   fault_hanging=norms_fh,
                                   fault_foot=norms_ff)

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

plt.plot(nodes[:,0],nodes[:,1],'ko')
plt.show()
# view nodes
#for i in boundary_indices:
#  plt.plot(nodes[:,0],nodes[:,1],'ko')
#  plt.plot(nodes[s[i],0],nodes[s[i],1],'bo')
#  plt.plot(nodes[s[i][0],0],nodes[s[i][0],1],'ro')
#  plt.show()

# find the optimal shape factor for each stencil. In 2D the optimal
# shape parameter tends to smallest one that yeilds a still invertible 
# Vandermonde matrix.  So the shape parameter is chosen so that the 
# Vandermonde matrix has a condition number of about 10^10.  This is 
# NOT an effective strategy in 3D because the Vandermonde matrix tends
# to be much better conditioned
eps = rbf.weights.shape_factor(nodes,s,basis,cond=cond,samples=200)

N = len(nodes)
modest.tic('forming G')

G = [[scipy.sparse.lil_matrix((N,N),dtype=np.float64) for mi in range(dim)] for di in range(dim)]
data = [np.zeros(N) for i in range(dim)]
# This can be parallelized!!!!
for di in range(dim):
  for mi in range(dim):
    # apply the PDE to interior nodes and free nodes
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

    # treat fault nodes as free nodes and the later reorganize the G
    # matrix so that the fault nodes are forces to be equal
    for i in ix['fault_hanging']+ix['fault_foot']:
      w = rbf_weight(nodes[i],
                 nodes[s[i]],
                 evaluate_coeffs_and_diffs(FreeBCOps[di][mi],i),
                 eps=eps[i],
                 Np=Np,
                 basis=basis,
                 cond=cond)
      G[di][mi][i,s[i]] = w

    # use the ghost node rows to enforce the free boundary conditions
    # at the boundary nodes
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
cs = ax.tripcolor(nodes[:,0],
                  nodes[:,1],
                  np.linalg.norm(out,axis=0),cmap=slip2)
plt.colorbar(cs)
plt.quiver(nodes[::2,0],nodes[::2,1],
           out[0,::2],out[1,::2],color='k',scale=20.0)



x = nodes[:,0]
z = nodes[:,1] - 10.0
y = 0 + 0*x
pnts = np.array([x,y,z]).T

#soln = okada.okada92(pnts,np.array([0.0,0.0,2.0]),np.array([0.0,-10.0,0.0]),20.0,1.0,0.0,np.pi/2.0)
#solnint = soln[ix['interior'],:]
#outint = out[:,ix['interior']].T
#print(np.max(np.abs(solnint[:,[0,1]]-outint)))
#plt.quiver(x[::2],z[::2]+10.0,soln[::2,0],soln[::2,2],color='b',scale=20.0)
#plt.show()


logging.basicConfig(level=logging.INFO)
summary()

plt.show()


