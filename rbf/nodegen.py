#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
import rbf.geometry
import rbf.normalize
import rbf.weights
import rbf.stencil
import modest
from modest import funtime
import logging
import random
logger = logging.getLogger(__name__)


def merge_nodes(**kwargs):
  out_dict = {}
  out_array = ()
  n = 0
  for k,a in kwargs.items():
    a = np.asarray(a,dtype=np.float64,order='c')
    idx = range(n,n+a.shape[0])
    n += a.shape[0]
    out_array += a,
    out_dict[k] = idx

  out_array = np.vstack(out_array)
  return out_array,out_dict


def normal(M):
  '''                                       
  returns the normal vector to the N-1 N-vectors  

  PARAMETERS
  ----------
    M: (N-1,N) array of vectors                      

  supports broadcasting    
  '''
  N = M.shape[-1]
  Msubs = [np.delete(M,i,-1) for i in range(N)]
  out = np.linalg.det(Msubs)
  out[1::2] *= -1
  out = np.rollaxis(out,-1)
  out /= np.linalg.norm(out,axis=-1)[...,None]
  return out
  
  
def bnd_intersection(inside,outside,vertices,simplices):
  dim = inside.shape[1]
  if dim == 2:
    out = rbf.geometry.cross_where_2d(inside,outside,vertices,simplices)
  if dim == 3:
    out = rbf.geometry.cross_where_3d(inside,outside,vertices,simplices)

  return out
  

def bnd_normal(inside,outside,vertices,simplices):
  dim = inside.shape[1]
  if dim == 2:
    out = rbf.geometry.cross_normals_2d(inside,outside,vertices,simplices)
  if dim == 3:
    out = rbf.geometry.cross_normals_3d(inside,outside,vertices,simplices)

  return out


def bnd_group(inside,outside,vertices,simplices,group):
  dim = inside.shape[1]
  if dim == 2:
    smp_ids = rbf.geometry.cross_which_2d(inside,outside,vertices,simplices)
  if dim == 3:
    smp_ids= rbf.geometry.cross_which_3d(inside,outside,vertices,simplices)

  out = np.array(group[smp_ids],copy=True)
  return out


def bnd_crossed(inside,outside,vertices,simplices):
  dim = inside.shape[1]
  if dim == 2:
    out = rbf.geometry.cross_count_2d(inside,outside,vertices,simplices)
  if dim == 3:
    out = rbf.geometry.cross_count_3d(inside,outside,vertices,simplices)

  return out

def bnd_contains(points,vertices,simplices):
  dim = points.shape[1]
  if dim == 2:
    out = rbf.geometry.contains_2d(points,vertices,simplices)
  if dim == 3:
    out = rbf.geometry.contains_3d(points,vertices,simplices)

  return out


def default_rho(p):
  return 1.0 + 0*p[:,0]


@modest.funtime  
def repel_step(free_nodes,
               fix_nodes=None,
               n=10,delta=0.1,
               rho=None):
  free_nodes = np.asarray(free_nodes,dtype=float)
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)
  if rho is None:
    rho = default_rho

  nodes = np.vstack((free_nodes,fix_nodes))
  i,d = rbf.stencil.nearest(free_nodes,nodes,n)
  i = i[:,1:]
  d = d[:,1:]
  c = 1.0/(rho(nodes)[i,None]*rho(free_nodes)[:,None,None])
  direction = np.sum(c*(free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  
  step = delta*d[:,0,None]*direction
  new_free_nodes = free_nodes + step
  return new_free_nodes


@modest.funtime
def repel_bounce(free_nodes,
                 vertices,
                 simplices,
                 fix_nodes=None,
                 itr=10,n=10,delta=0.1,
                 rho=None,max_bounces=3):
  '''
  nodes are repelled by eachother and bounce off boundaries
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)
  if rho is None:
    rho = default_rho

  # this is used for the lengthscale of the domain
  scale = np.max(vertices) - np.min(vertices)

  # ensure that the number of nodes used to determine repulsion force
  # is less than the total number of nodes
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_new = repel_step(free_nodes,
                                fix_nodes=fix_nodes,
                                n=n,delta=delta,rho=rho)

    # boolean array of nodes which are now outside the domain
    crossed = ~bnd_contains(free_nodes_new,vertices,simplices)
    bounces = 0
    while np.any(crossed):
      # point where nodes intersected the boundary
      inter = bnd_intersection(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)

      # normal vector to intersection point
      norms = bnd_normal(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)
      
      # distance that the node wanted to travel beyond the boundary
      res = free_nodes_new[crossed] - inter

      # move the previous node position to just within the boundary
      free_nodes[crossed] = inter - 1e-10*scale*norms

      # 3 is the number of bounces allowed   
      if bounces > max_bounces:
        free_nodes_new[crossed] = inter - 1e-10*scale*norms
        break

      else: 
        # bouce node off the boundary
        free_nodes_new[crossed] -= 2*norms*np.sum(res*norms,1)[:,None]        
        # check to see if the bounced node is now within the domain, 
        # if not then iterations continue
        crossed = ~bnd_contains(free_nodes_new,vertices,simplices)
        bounces += 1

    free_nodes = free_nodes_new  
  
  return free_nodes

@modest.funtime
def repel_stick(free_nodes,
                vertices,
                simplices,
                groups=None, 
                fix_nodes=None,
                itr=10,n=10,delta=0.1,
                rho=None,max_bounces=3):
  '''
  nodes are repelled by eachother and then become fixed when they hit 
  a boundary
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 
  if groups is None:
    groups = np.ones(simplices.shape[0],dtype=int)

  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)
  if rho is None:
    rho = default_rho

  # these arrays will be populated 
  node_norm = np.zeros(free_nodes.shape,dtype=float)
  node_group = np.zeros(free_nodes.shape[0],dtype=int)

  # length scale of the domain
  scale = np.max(vertices) - np.min(vertices)

  # ensure that the number of nodes used to compute repulsion force is
  # less than or equal to the total number of nodes
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    # indices of all nodes not associated with a group (i.e. free 
    # nodes)
    ungrouped = np.nonzero(node_group==0)[0]

    # indices of nodes associated with a group (i.e. nodes which 
    # intersected a boundary
    grouped = np.nonzero(node_group!=0)[0]

    # nodes which are stationary 
    grouped_free_nodes = free_nodes[grouped]    
    all_fix_nodes = np.vstack((fix_nodes,grouped_free_nodes))

    # nodes which are free
    ungrouped_free_nodes = free_nodes[ungrouped]

    # new position of free nodes
    ungrouped_free_nodes_new = repel_step(
                                 ungrouped_free_nodes,
                                 fix_nodes=all_fix_nodes,
                                 n=n,delta=delta,rho=rho)

    # indices if free nodes which crossed a boundary
    crossed = ~bnd_contains(ungrouped_free_nodes_new,vertices,simplices)


    # if a node intersected a boundary then associate it with a group
    node_group[ungrouped[crossed]] = bnd_group(
                                       ungrouped_free_nodes[crossed],     
                                       ungrouped_free_nodes_new[crossed], 
                                       vertices,simplices,groups)

    # if a node intersected a boundary then associate it with a normal
    node_norm[ungrouped[crossed]] = bnd_normal(
                                      ungrouped_free_nodes[crossed],     
                                      ungrouped_free_nodes_new[crossed], 
                                      vertices,simplices)

    # intersection point for nodes which crossed a boundary
    inter = bnd_intersection(
              ungrouped_free_nodes[crossed],     
              ungrouped_free_nodes_new[crossed],
              vertices,simplices)

    # new position of nodes which crossed the boundary is just within
    # the intersection point
    ungrouped_free_nodes_new[crossed] = inter - 1e-10*scale*node_norm[ungrouped[crossed]]
    free_nodes[ungrouped] = ungrouped_free_nodes_new

  return free_nodes,node_norm,node_group


@modest.funtime
def generate_nodes_1d(N,lower,upper,
                      rho=None,itr=100,n=10,delta=0.01,
                      normalize_rho=True):
  if n > N:
    n = N

  if N < 2:
    nodes = np.array([[lower],[upper]])
    return nodes[:N]

  if rho is None:
    rho = default_rho

  if normalize_rho:
    rho = rbf.normalize.normalize_1d(rho,lower,upper,by='max')

  max_sample_size = 100*N
  H = rbf.halton.Halton(2)
  nodes = np.zeros((0,1))
  cnt = 0
  acceptance = 1.0
  while nodes.shape[0] < (N-2):
    if acceptance == 0.0:
      sample_size = max_sample_size
    else:
      sample_size = int(((N-2)-nodes.shape[0])/acceptance) + 1
      if sample_size > max_sample_size:
        sample_size = max_sample_size

    cnt += sample_size
    seq = H(sample_size)
    seq1 = seq[:,[0]]
    seq2 = seq[:,1]
    new_nodes = (upper-lower)*seq1 + lower
    new_nodes = new_nodes[rho(new_nodes) > seq2]
    nodes = np.vstack((nodes,new_nodes))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:(N-2)]
  for i in range(itr):
    nodes = repel_step(nodes,fix_nodes=np.array([[lower],[upper]]),
                       n=n,delta=delta,rho=rho)

  nodes = np.vstack(([[lower],[upper]],nodes))
  nodes = np.sort(nodes,axis=0)
  return nodes


@modest.funtime
def generate_nodes(N,vertices,simplices,groups=None,fix_nodes=None,rho=None,
                   itr=20,n=10,delta=0.1,normalize_rho=True):

  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 
  if groups is None:
    groups = np.ones(simplices.shape[0],dtype=int)

  groups = np.asarray(groups,dtype=int) 

  if rho is None:
    rho = default_rho

  if normalize_rho:
    rho = rbf.normalize.normalize(rho,vertices,simplices,by='max')

  lb = np.min(vertices,0)
  ub = np.max(vertices,0)
  max_sample_size = 100*N  
  ndim = lb.shape[0]
  H = rbf.halton.Halton(ndim+1)
  nodes = np.zeros((0,ndim))
  cnt = 0
  acceptance = 1.0
  while nodes.shape[0] < N:
    if acceptance == 0.0:
      sample_size = max_sample_size    
    else:
      sample_size = int((N-nodes.shape[0])/acceptance) + 1
      if sample_size > max_sample_size:
        sample_size = max_sample_size

    cnt += sample_size
    seqNd = H(sample_size)
    seq1d = seqNd[:,-1]
    new_nodes = (ub-lb)*seqNd[:,:ndim] + lb
    new_nodes = new_nodes[rho(new_nodes) > seq1d]

    new_nodes = new_nodes[bnd_contains(new_nodes,vertices,simplices)]
    nodes = np.vstack((nodes,new_nodes))
    logger.info('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]
  logger.info('repelling nodes with boundary bouncing') 
  nodes = repel_bounce(nodes,vertices,simplices,fix_nodes=fix_nodes,itr=itr,
                       n=n,delta=delta,rho=rho)

  logger.info('repelling nodes with boundary sticking') 
  nodes,norms,grp = repel_stick(nodes,vertices,simplices,groups,
                                fix_nodes=fix_nodes,
                                itr=itr,n=n,
                                delta=delta,rho=rho)

  return nodes,norms,grp






    

  
