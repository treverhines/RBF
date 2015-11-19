#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
import modest
from modest import funtime
import logging
logger = logging.basicConfig(level=logging.INFO)


def _boundary_filter(nodes,bnd=None):
  if bnd == None:
    return nodes
  else:
    return nodes[bnd(nodes)]


def _density_filter(nodes,seq,rho=None):
  if rho == None:
    return nodes
  else:
    return nodes[rho(nodes) > seq]


def _proximity_filter(nodes,seq,rho=None):
  if rho == None:
    return nodes
  else:
    return nodes[rho(nodes) > seq]


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
  
  
def bnd_adjacent(pnt,Nadj,bnd,eps=1e-4,Nsmp=None):
  '''                                 
  returns a Npnt x Nadj x Ndim array         
  '''
  Npnt,Ndim = pnt.shape
  if Nsmp is None:
    Nsmp = Nadj*10

  smp = pnt[:,None,:] + eps*rbf.halton.halton(Nsmp,Ndim) - eps/2
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
  adj = bnd_intersection(pin,pout,bnd,tol=eps*1e-6)
  adj = adj.reshape((Npnt,Nadj,Ndim))
  return adj


def bnd_normal(pnt,bnd,**kwargs):
  eps = kwargs.pop('eps',1e-4)
  Ndim = pnt.shape[1]
  adj = bnd_adjacent(pnt,Ndim,bnd,eps=eps,**kwargs)
  adj = adj.reshape((pnt.shape[0],Ndim,Ndim))
  adj -= adj[:,[0],:]
  adj = adj[:,1:,:]
  norm = normal(adj)
  is_inside = bnd(pnt + eps*norm)
  norm[is_inside] *= -1
  return norm
  

def _repel_k(free_nodes,fix_nodes,n,eps):
  tol = 1e-10
  nodes = np.vstack((free_nodes,fix_nodes))
  T = scipy.spatial.cKDTree(nodes)
  d,i = T.query(free_nodes,n)
  i = i[:,1:]
  d = d[:,1:]
  force = np.sum((free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
  mag = np.linalg.norm(force,axis=1)
  idx = mag > tol
  force[idx] /= mag[idx,None]
  force[~idx] *= 0.0
  step = eps*d[:,0,None]*force
  free_nodes = free_nodes + step
  return free_nodes


def _repel(free_nodes,fix_nodes=None,itr=10,n=10,eps=0.1,bnd=None):
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes)
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_old = free_nodes
    free_nodes = _repel_k(free_nodes,fix_nodes,n,eps)
    if bnd is not None:
      is_outside = ~bnd(free_nodes)
      c = 0
      while np.any(is_outside):
        bnd_nodes = bnd_intersection(
                      free_nodes_old[is_outside],     
                      free_nodes[is_outside],
                      bnd)
        norms = bnd_normal(bnd_nodes,bnd)
        res = free_nodes[is_outside] - bnd_nodes
        free_nodes_old[is_outside] = bnd_nodes 
        # 3 is the number of bounces allowed   
        if c > 3:
          free_nodes[is_outside] = bnd_nodes
          break

        else: 
          free_nodes[is_outside] -= 2*norms*np.sum(res*norms,1)[:,None]        
          is_outside = ~bnd(free_nodes)

        c += 1

  return free_nodes


def _repel_and_stick(free_nodes,fix_nodes=None,itr=10,n=10,eps=0.1,bnd=None):
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes)
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_old = free_nodes
    free_nodes = _repel_k(free_nodes,fix_nodes,n,eps)
    if bnd is not None:
      is_outside = ~bnd(free_nodes)
      bnd_nodes = bnd_intersection(
                    free_nodes_old[is_outside],     
                    free_nodes[is_outside],
                    bnd)
      fix_nodes = np.vstack((fix_nodes,bnd_nodes))
      free_nodes = free_nodes[~is_outside] 


  return free_nodes,fix_nodes


def pick_nodes(N,lb,ub,bnd_nodes=None,bnd=None,rho=None,
               repel_itr=10,repel_n=10,repel_eps=0.1):
  lb = np.asarray(lb)
  ub = np.asarray(ub)
  assert len(lb) == len(ub)
  assert np.all(ub > lb)
  ndim = len(lb)
  H = rbf.halton.Halton(ndim+1)
  nodes = np.zeros((0,ndim))
  cnt = 0
  acceptance = 1.0
  while nodes.shape[0] < N:
    sample_size = int((N-nodes.shape[0])/acceptance) + 1
    cnt += sample_size
    seqNd = H(sample_size)
    seq1d = seqNd[:,-1]
    new_nodes = (ub-lb)*seqNd[:,:ndim] + lb
    new_nodes = _density_filter(new_nodes,seq1d,rho)
    nodes = np.vstack((nodes,new_nodes))
    nodes = _boundary_filter(nodes,bnd)    
    acceptance = nodes.shape[0]/cnt
    if acceptance <= 0:
      print('Warning: no samples are being retained')

  nodes = nodes[:N]
  nodes = _repel(nodes,bnd_nodes,
                 itr=repel_itr,
                 n=repel_n,
                 eps=repel_eps,
                 bnd=bnd)

  nodes,bnd_nodes = _repel_and_stick(
                      nodes,bnd_nodes,
                      itr=repel_itr,
                      n=repel_n,
                      eps=repel_eps,
                      bnd=bnd)

  return nodes,bnd_nodes






    

  
