#!/usr/bin/env python
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
import rbf.geometry
import modest
from modest import funtime
import logging
logger = logging.getLogger(__name__)


def mcint(f,vert,smp,N=10000):
  lb = np.min(vert,0)
  ub = np.max(vert,0)
  dim = lb.shape[0]
  pnts = rbf.halton.halton(N,dim)*(ub-lb) + lb
  val = rho(pnts)
  if dim == 2:
    val *= rbf.geometry.contains_2d(pnts,vert,smp).astype(float)
  if dim == 3:
    val *= rbf.geometry.contains_3d(pnts,vert,smp).astype(float)
  soln = np.mean(val)*np.prod(ub-lb)
  return soln


def normalize(fin,vert,smp):
  integral = mcint(fin,vert,smp)
  def fout(p):
    return fin(p)/integral

  return fout


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


def distance(pnt,pnts,vert,smp):
  '''
  returns euclidean distance between pnt and pnts. If the line segment
  between pnt and pnts crosses a boundary then the distance is inf
  '''  
  pnt = np.repeat(pnt[None,:],pnts.shape[0],axis=0)
  cc = rbf.geometry.cross_count_2d(pnt,pnts,vert,smp)
  dist = np.sqrt(np.sum((pnts-pnt)**2,1))
  dist[cc>0] = np.inf
  return dist

def stencilate(pnts,N,vert,smp):
  M = pnts.shape[0]
  T = scipy.spatial.cKDTree(pnts)
  dist,neighbors= T.query(pnts,N+1)
  neighbors = neighbors[:,1:]
  mindist = dist[:,1]
  for i in range(M):
    # distance from point i to nearest neighbors, crossing
    # a boundary gives infinite distance
    dist_i = distance(pnts[i],pnts[neighbors[i]],vert,smp)
    query_size = N+1
    while np.any(np.isinf(dist_i)):
      # if some neighbors cross a boundary then query a larger
      # set of nearest neighbors from the KDTree
      query_size += N
      dist_i,neighbors_i = T.query(pnts[i],query_size)
      neighbors_i = neighbors_i[1:]
      # recompute distance to larger set of neighbors
      dist_i = distance(pnts[i],pnts[neighbors_i],vert,smp)
      # assign the closest N neighbors to the neighbors array
      neighbors[i] = neighbors_i[np.argsort(dist_i)[:N]]
      dist_i = dist_i[np.argsort(dist_i)[:N]]
      mindist[i] = dist_i[0]
      if query_size >= (M-N):
        print('WARNING: could not find %s nearest neighbors for point '
              '%s without crossing a boundary' % (N,pnts[i]))
        break

  return neighbors,mindist


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

  
def _repel_step(free_nodes,fix_nodes,n,delta,rho):
  nodes = np.vstack((free_nodes,fix_nodes))
  T = scipy.spatial.cKDTree(nodes)
  d,i = T.query(free_nodes,n)
  i = i[:,1:]
  d = d[:,1:]
  c = 1.0/rho(nodes)[i,None]*rho(free_nodes)[:,None,None]  
  direction = np.sum(c*(free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  
  step = delta*d[:,0,None]*direction
  new_free_nodes = free_nodes + step
  return new_free_nodes


def default_rho(p):
  return 1.0 + 0*p[:,0]

def repel_bounce(free_nodes,
                 vertices,
                 simplices,
                 fix_nodes=None,
                 itr=10,n=10,delta=0.1,
                 rho=None,max_bounces=3):
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  dim = free_nodes.shape[1]
  if rho is None:
    rho = default_rho

  if fix_nodes is None:
    fix_nodes = np.zeros((0,dim))

  fix_nodes = np.asarray(fix_nodes,dtype=float)
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    free_nodes_new = _repel_step(free_nodes,
                                 fix_nodes,n,delta,rho)
    crossed = ~bnd_contains(free_nodes_new,vertices,simplices)
    bounces = 0
    while np.any(crossed):
      inter = bnd_intersection(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)
      norms = bnd_normal(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)
      res = free_nodes_new[crossed] - inter
      free_nodes[crossed] = inter - 1e-10*norms
      # 3 is the number of bounces allowed   
      if bounces > max_bounces:
        free_nodes_new[crossed] = inter
        break

      else: 
        free_nodes_new[crossed] -= 2*norms*np.sum(res*norms,1)[:,None]        
        crossed = ~bnd_contains(free_nodes_new,vertices,simplices)
        bounces += 1

    free_nodes = free_nodes_new  
  
  return free_nodes


def repel_stick(free_nodes,
                vertices,
                simplices,
                groups, 
                fix_nodes=None,
                itr=10,n=10,delta=0.1,
                rho=None,max_bounces=3):

  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  norm = np.zeros(free_nodes.shape,dtype=float)
  grp = np.zeros(free_nodes.shape[0],dtype=int)
  dim = free_nodes.shape[1]
  if rho is None:
    rho = default_rho

  if fix_nodes is None:
    fix_nodes = np.zeros((0,dim))

  fix_nodes = np.asarray(fix_nodes,dtype=float)
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    ungrouped = np.nonzero(grp==0)[0]
    grouped = np.nonzero(grp!=0)[0]
    grouped_free_nodes = free_nodes[grouped]    
    all_fix_nodes = np.vstack((fix_nodes,grouped_free_nodes))
    ungrouped_free_nodes = free_nodes[ungrouped]
    ungrouped_free_nodes_new = _repel_step(
                                 ungrouped_free_nodes,
                                 all_fix_nodes,n,delta,rho)
    crossed = ~bnd_contains(ungrouped_free_nodes_new,vertices,simplices)
    inter = bnd_intersection(
              ungrouped_free_nodes[crossed],     
              ungrouped_free_nodes_new[crossed],
              vertices,simplices)
    grp[ungrouped[crossed]] = bnd_group(
                                ungrouped_free_nodes[crossed],     
                                ungrouped_free_nodes_new[crossed], 
                                vertices,simplices,groups)
    norm[ungrouped[crossed]] = bnd_normal(
                                ungrouped_free_nodes[crossed],     
                                ungrouped_free_nodes_new[crossed], 
                                vertices,simplices)
    ungrouped_free_nodes_new[crossed] = inter
    free_nodes[ungrouped] = ungrouped_free_nodes_new

  return free_nodes,norm,grp


def generate_nodes(N,vertices,simplices,groups,fix_nodes=None,rho=None,
                   itr=20,n=10,delta=0.1):
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
    if rho is not None:
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






    

  
