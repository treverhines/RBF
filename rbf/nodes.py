#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
import modest
from modest import funtime
import logging
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


def stencilate(x,N):
  T = scipy.spatial.cKDTree(x)
  d,i = T.query(x,N)
  if N == 1:
    i = i[:,None]

  return i

def mindist(x):
  T = scipy.spatial.cKDTree(x)
  d,i = T.query(x,2)
  return d[:,1]


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
  
  
def bnd_intersection(inside,outside,bnd,tol=1e-10):
  inside = np.copy(inside)
  outside = np.copy(outside)
  err = np.linalg.norm(inside-outside,axis=1)
  while np.any(err > tol):
    mid = (inside + outside)/2
    result = bnd(mid)
    inside[result] = mid[result]
    outside[~result] = mid[~result]
    err = np.linalg.norm(inside-outside,axis=1)
  
  return inside
  
  
def bnd_adjacent(pnt,Nadj,bnd,rho=1e-4,Nsmp=None):
  '''                                 
  returns a Npnt x Nadj x Ndim array          
  '''
  Npnt,Ndim = pnt.shape
  if Nsmp is None:
    Nsmp = Nadj*10

  smp = pnt[:,None,:] + rho*rbf.halton.halton(Nsmp,Ndim) - rho/2
  smp = smp.reshape((Npnt*Nsmp,Ndim))
  is_inside = bnd(smp)
  smp = smp.reshape((Npnt,Nsmp,Ndim))
  is_inside = is_inside.reshape((Npnt,Nsmp))
  pin = []
  pout = []
  for i in range(Npnt):
    pin += [smp[i,is_inside[i]][:Nadj]]
    pout += [smp[i,~is_inside[i]][:Nadj]]
    assert (len(pin[-1]) == Nadj & len(pout[-1]) == Nadj), (
      'Not enough inside and/or outside points found to generate '
      'requested number of adjacent boundary points')

  pin = np.reshape(pin,(Npnt*Nadj,Ndim))
  pout = np.reshape(pout,(Npnt*Nadj,Ndim))
  adj = bnd_intersection(pin,pout,bnd,tol=rho*1e-6)
  adj = adj.reshape((Npnt,Nadj,Ndim))
  return adj


def bnd_normal(pnt,bnd,**kwargs):
  rho = kwargs.pop('rho',1e-4)
  Ndim = pnt.shape[1]
  adj = bnd_adjacent(pnt,Ndim,bnd,rho=rho,**kwargs)
  adj = adj.reshape((pnt.shape[0],Ndim,Ndim))
  adj -= adj[:,[0],:]
  adj = adj[:,1:,:]
  norm = normal(adj)
  is_inside = bnd(pnt + rho*norm)
  norm[is_inside] *= -1
  return norm
  

def _repel_k(free_nodes,fix_nodes,n,delta):
  tol = 1e-10
  nodes = np.vstack((free_nodes,fix_nodes))
  T = scipy.spatial.cKDTree(nodes)
  d,i = T.query(free_nodes,n)
  i = i[:,1:]
  d = d[:,1:]
  direction = np.sum((free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  
  step = delta*d[:,0,None]*direction
  new_free_nodes = free_nodes + step
  return new_free_nodes


def repel_bounce(free_nodes,
                 fix_nodes=None,
                 itr=10,n=10,delta=0.1,
                 bnd=None,max_bounces=3):
  free_nodes = np.asarray(free_nodes,dtype=np.float64,order='c')
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=np.float64,order='c')
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_old = free_nodes
    free_nodes = _repel_k(free_nodes,fix_nodes,n,delta)
    if bnd is not None:
      is_outside = ~bnd(free_nodes)
      bounces = 0
      while np.any(is_outside):
        bnd_nodes = bnd_intersection(
                      free_nodes_old[is_outside],     
                      free_nodes[is_outside],
                      bnd)
        bnd_norms = bnd_normal(bnd_nodes,bnd)
        res = free_nodes[is_outside] - bnd_nodes
        free_nodes_old[is_outside] = bnd_nodes 
        # 3 is the number of bounces allowed   
        if bounces > max_bounces:
          free_nodes[is_outside] = bnd_nodes
          break

        else: 
          free_nodes[is_outside] -= 2*bnd_norms*np.sum(res*bnd_norms,1)[:,None]        
          is_outside = ~bnd(free_nodes)

        bounces += 1

  return free_nodes


def repel_stick(free_nodes,
                fix_nodes=None,
                itr=10,n=10,delta=0.1,
                bnd=None):
  free_nodes = np.asarray(free_nodes,dtype=np.float64,order='c')
  # fix_nodes will contain all nodes that are initially fixed 
  # and ones which intersected the boundary
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]),
                         dtype=np.float64,order='c')

  fix_nodes = np.asarray(fix_nodes,dtype=np.float64,order='c')

  # bnd_nodes will contain only nodes that intersected the
  # boundary 
  bnd_nodes = np.zeros((0,free_nodes.shape[1]),
                       dtype=np.float64,order='c')

  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_old = free_nodes
    free_nodes = _repel_k(free_nodes,fix_nodes,n,delta)
    if bnd is not None:
      is_outside = ~bnd(free_nodes)
      intersect = bnd_intersection(
                    free_nodes_old[is_outside],     
                    free_nodes[is_outside],
                    bnd)
      fix_nodes = np.vstack((fix_nodes,intersect))
      bnd_nodes = np.vstack((bnd_nodes,intersect))
      free_nodes = free_nodes[~is_outside] 


  return free_nodes,bnd_nodes


def generate_nodes(N,lb,ub,bnd,fix_nodes=None,rho=None,
                   itr=20,n=10,delta=0.1):
  max_sample_size = 100*N  
  lb = np.asarray(lb)
  ub = np.asarray(ub)
  assert lb.shape[0] == ub.shape[0]
  assert np.all(ub > lb)
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
    if rho is not None:
      new_nodes = new_nodes[rho(new_nodes) > seq1d]

    new_nodes = new_nodes[bnd(new_nodes)]
    nodes = np.vstack((nodes,new_nodes))
    logger.info('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]
  logger.info('repelling nodes with boundary bouncing') 
  nodes = repel_bounce(nodes,fix_nodes,itr=itr,
                        n=n,delta=delta,bnd=bnd)

  logger.info('repelling nodes with boundary sticking') 
  nodes,bnd_nodes = repel_stick(nodes,fix_nodes,
                                itr=itr,n=n,
                                delta=delta,bnd=bnd)

  logger.info('computing boundary normals')
  bnd_norms = bnd_normal(bnd_nodes,bnd)
  return nodes,bnd_nodes,bnd_norms






    

  
