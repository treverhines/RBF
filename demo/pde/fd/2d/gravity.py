#!/usr/bin/env python
import numpy as np
import rbf.nodes
from rbf.basis import phs5 as basis
from rbf.integrate import density_normalizer
from rbf.geometry import contains
from rbf.fd import weight_matrix
import rbf.stencil
import myplot.cm
from rbf.halton import Halton
from rbf.formulation import symbolic_coeffs_and_diffs,coeffs_and_diffs
from rbf.formulation import evaluate_coeffs
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.sparse
import scipy.sparse.linalg
import logging
import sympy as sp
logging.basicConfig(level=logging.INFO)
import sys

def solver(G,d):
  out = scipy.sparse.spsolve(G,d)
  return out
  
# formulate the PDE
x = sp.symbols('x0:2')
n = sp.symbols('n0:2')
u = (sp.Function('u0')(*x),
     sp.Function('u1')(*x))
#L = sp.Function('lambda')(*x)
#M = sp.Function('mu')(*x)
L = sp.symbols('lambda')
M = sp.symbols('mu')
dim = 2
F = [[u[i].diff(x[j]) for i in range(dim)] for j in range(dim)]
F = sp.Matrix(F)
strain = sp.Rational(1,2)*(F + F.T)
stress = L*sp.eye(dim)*sp.trace(strain) + 2*M*strain
PDEs = [sum(stress[i,j].diff(x[j]) for j in range(dim)) for i in range(dim)]
FreeBCs = [sum(stress[i,j]*n[j] for j in range(dim)) for i in range(dim)]
FixBCs = [u[i] for i in range(dim)]

# norms is an array which is later defined
sym2num = {L:1.0,
           M:1.0,
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

N = 2000
Ns = 30
order = 2

# domain vertices
surf_vert_x = np.linspace(-5,5,200)
surf_vert_y = 1.0/(1 + 1.0*(surf_vert_x)**2)
#surf_vert_y = 2.0*np.sin(4*np.pi*surf_vert_x/10.0)

surf_vert = np.array([surf_vert_x,surf_vert_y+10]).T
surf_smp = np.array([np.arange(199),np.arange(1,200)]).T
bot_vert = np.array([[-5.0,-5.0],
                     [5.0,-5.0]])
top_vert = np.array([[5.0,11.5],[-5.0,11.5]])
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

nodes_d,sidx = rbf.nodes.menodes(N,vert,smp,rho=rho)
norms_d = simplex_normals[sidx]
group_d = grp[sidx]
group_d[sidx==-1] = 0

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


