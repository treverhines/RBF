#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.halton
from rbf.geometry import (boundary_intersection,
                          boundary_normal,
                          boundary_group,
                          boundary_contains,
                          boundary_cross_count,
                          is_valid)
import rbf.normalize
import rbf.stencil
from modest import funtime
import logging
from itertools import combinations
logger = logging.getLogger(__name__)


def verify_node_spacing(rho,nodes,tol=0.25):
  '''
  Returns indices of nodes which are consistent with the node
  density function rho. Indexing nodes with the output 
  will return a pared down node set that is consistent with rho
  '''
  logger.info('verifying node spacing')  
  dim = nodes.shape[1]
  valid_indices = np.arange(nodes.shape[0],dtype=int)
  # minimum distance allowed between nodes
  mindist = tol/(rho(nodes)**(1.0/dim))
  s,dx = rbf.stencil.nearest(nodes,nodes,2)
  while np.any(dx[:,1] < mindist):
    toss = np.argmin(dx[:,1]/mindist)
    logger.info('removing node %s for being too close to adjacent node' 
                % valid_indices[toss])
    nodes = np.delete(nodes,toss,axis=0)
    valid_indices = np.delete(valid_indices,toss,axis=0)
    mindist = tol/(rho(nodes)**(1.0/dim))
    s,dx = rbf.stencil.nearest(nodes,nodes,2)

  return valid_indices


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


def default_rho(p):
  return 1.0 + 0*p[:,0]


def repel_step(free_nodes,
               fix_nodes=None,
               n=3,delta=0.1,
               rho=None):  
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  # if n is 0 or 1 then the nodes remain stationary
  if n <= 1:
    return free_nodes

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


def repel_bounce(free_nodes,
                 vertices,
                 simplices,
                 fix_nodes=None,
                 itr=50,n=3,delta=0.1,
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
    crossed = ~boundary_contains(free_nodes_new,vertices,simplices)
    bounces = 0
    while np.any(crossed):
      # point where nodes intersected the boundary
      inter = boundary_intersection(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)

      # normal vector to intersection point
      norms = boundary_normal(
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
        crossed = ~boundary_contains(free_nodes_new,vertices,simplices)
        bounces += 1

    free_nodes = free_nodes_new  
  
  return free_nodes


def repel_stick(free_nodes,
                vertices,
                simplices,
                groups=None, 
                fix_nodes=None,
                itr=50,n=3,delta=0.1,
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
    crossed = ~boundary_contains(ungrouped_free_nodes_new,vertices,simplices)
  
    # if a node intersected a boundary then associate it with a group
    node_group[ungrouped[crossed]] = boundary_group(
                                       ungrouped_free_nodes[crossed],     
                                       ungrouped_free_nodes_new[crossed], 
                                       vertices,simplices,groups)

    # if a node intersected a boundary then associate it with a normal
    node_norm[ungrouped[crossed]] = boundary_normal(
                                      ungrouped_free_nodes[crossed],     
                                      ungrouped_free_nodes_new[crossed], 
                                      vertices,simplices)

    # intersection point for nodes which crossed a boundary
    inter = boundary_intersection(
              ungrouped_free_nodes[crossed],     
              ungrouped_free_nodes_new[crossed],
              vertices,simplices)

    # new position of nodes which crossed the boundary is just within
    # the intersection point
    ungrouped_free_nodes_new[crossed] = inter - 1e-10*scale*node_norm[ungrouped[crossed]]
    free_nodes[ungrouped] = ungrouped_free_nodes_new

  return free_nodes,node_norm,node_group


@funtime
def volume(rho,vertices,simplices,groups=None,fix_nodes=None,
           itr=50,n=3,delta=0.1):
  '''
  Generates nodes within the N-dimensional volume enclosed by the 
  simplexes using a minimum energy algorithm.  At each iteration 
  the nearest neighbors to each node are found and then a repulsion
  force is calculated using the distance to the nearest neighbors and
  their charges (which is inversely proportional to the node density).
  Each node then moves in the direction of the net force acting on it.  
  The step size is equal to delta times the distance to the nearest 
  node.  

  Paramters
  ---------
    rho: node density function. Takes a (M,N) array of coordinates
      in N dimensional space and returns an (M,) array of node
      densities at those coordinates 

    vertices: boundary vertices

    simplices: describes how the vertices are connected to form the 
      boundary
    
    groups (default=None): Array of integers identifying groups that
      the simplexes belong to. This is used to identify nodes
      belonging to particular boundaries
  
    fix_nodes (default=None): Nodes which do not move and only provide
      a repulsion force
 
    itr (default=100): number of repulsion iterations.  If this number
      is small then the nodes will not reach a minimum energy
      equilibrium.

    n (default=3): number of neighboring nodes to use when calculating
      repulsion force. When n is small, the equilibrium state tends to
      be a uniform node distribution (regardless of the specified
      rho), when n is large, nodes tend to get pushed up against the
      boundaries.  It is best to use a small n and a value for itr
      which is large enough for the nodes to disperse a little bit but
      not reach equilibrium.

    delta (default=0.1): Controls the node step size for each
      iteration.  The tep size is equal to delta times the distance to
      the nearest neighbor
  ''' 
  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 

  if not is_valid(simplices):
    print(
      'WARNING: One or more simplexes do not share an edge with '
      'another simplex which may indicate that the specified boundary '
      'is not closed. ')
    logger.warning(
      'One or more simplexes do not share an edge with another simplex '
      'which may indicate that the specified boundary is not closed. ')

  N,err,minval,maxval = rbf.normalize.rmcint(rho,vertices,simplices)
  assert minval >= 0.0, (
    'values in node density function must be positive')
  
  N = int(np.round(N))

  def rho_normalized(p):
    return rho(p)/maxval

  if groups is None:
    groups = np.ones(simplices.shape[0],dtype=int)

  groups = np.asarray(groups,dtype=int) 

  lb = np.min(vertices,0)
  ub = np.max(vertices,0)
  max_sample_size = 1000000
  ndim = lb.shape[0]
  H = rbf.halton.Halton(ndim+1)
  nodes = np.zeros((0,ndim))
  cnt = 0
  acceptance = 1.0
  while nodes.shape[0] < N:
    if acceptance == 0.0:
      sample_size = max_sample_size    
    else:
      sample_size = int(np.ceil((N-nodes.shape[0])/acceptance))
      if sample_size > max_sample_size:
        sample_size = max_sample_size

    cnt += sample_size
    seqNd = H(sample_size)
    seq1d = seqNd[:,-1]
    new_nodes = (ub-lb)*seqNd[:,:ndim] + lb
    new_nodes = new_nodes[rho_normalized(new_nodes) > seq1d]

    new_nodes = new_nodes[boundary_contains(new_nodes,vertices,simplices)]
    nodes = np.vstack((nodes,new_nodes))
    logger.info('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]
  logger.info('repelling nodes with boundary bouncing') 

  nodes = repel_bounce(nodes,vertices,simplices,fix_nodes=fix_nodes,itr=itr,
                       n=n,delta=delta,rho=rho_normalized)

  logger.info('repelling nodes with boundary sticking') 
  nodes,norms,grp = repel_stick(nodes,vertices,simplices,groups,
                                fix_nodes=fix_nodes,
                                itr=itr,n=n,
                                delta=delta,rho=rho_normalized)

  idx = verify_node_spacing(rho,nodes)
  nodes = nodes[idx]
  norms = norms[idx]
  grp = grp[idx]

  return nodes,norms,grp


def simplex_rotation(vert):
  '''                                                                                      
  returns a matrix that rotates the simplex such that      
  its normal is pointing in the direction of the last axis    
  '''
  vert = np.asarray(vert)
  dim = vert.shape[1]
  if dim == 2:
    # anchor one node of the simplex at the origin      
    v1 = vert[1,:] - vert[0,:]
    # find the normal vector to the simplex         
    normal = rbf.geometry.normal(np.array([v1]))
    # find the angle between the y axis and the simplex normal   
    argz = np.arctan2(normal[0],normal[1])
    # create a rotation matrix that rotates the normal onto the y axis
    R = np.array([[np.cos(argz), -np.sin(argz)],
                  [np.sin(argz),  np.cos(argz)]])
    return R

  if dim == 3:
    # anchor one node of the simplex at the origin 
    v1 = vert[1,:] - vert[0,:]
    v2 = vert[2,:] - vert[0,:]
    # find the normal vector to the simplex      
    normal = rbf.geometry.normal(np.array([v1,v2]))

    # project the normal onto to y-z plane and then compute     
    # the angle between the normal and the z axis      
    argx = np.arctan2(normal[1],normal[2])

    # rotate about x axis                
    R1 = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(argx), -np.sin(argx)],
                   [0.0, np.sin(argx),  np.cos(argx)]])

    # rotate normal by the above angle so that the normal  
    # is now on the x-z plane                          
    normal = R1.dot(normal)

    # find the angle between rotated normal and the z axis 
    argy = np.arctan2(normal[0],normal[2])

    # rotate about the y axis                        
    R2 = np.array([[np.cos(argy), 0.0,-np.sin(argy)],
                   [0.0, 1.0, 0.0],
                   [np.sin(argy), 0.0, np.cos(argy)]])

    # this matrix product takes the original normal and rotates it
    # onto the z axis                   
    R = R2.dot(R1)
    return R


def nodes_on_simplex(rho,vert,fix_nodes=None,**kwargs):
  '''                                                                                      
  This finds nodes on the given simplex which would be consistent 
  with the given node density function rho.    
                                                                                           
  '''
  r,c = np.shape(vert)
  assert r == c, (
    'number of vertices given should be the equal to the number of '
    'spatial dimensions')

  vert = np.asarray(vert)
  dim = vert.shape[1]

  if fix_nodes is None:
    fix_nodes = np.zeros((0,dim))

  # find the rotation matrix which rotates the simplex normal onto the
  # last axis                             
  R = simplex_rotation(vert)

  # _r denotes rotated values           
  vert_r = np.einsum('ij,...j->...i',R,vert)
  const_r = vert_r[0,-1]

  # remove last dimension from the vertices  
  vert_r = vert_r[:,:-1]

  # _r denotes rotated values               
  fix_r = np.einsum('ij,...j->...i',R,fix_nodes)

  # remove last dimension from the vertices       
  fix_r = fix_r[:,:-1]

  # define a facet node density function, which takes positions in the
  # rotated coordinate system, rotates them back to the original  
  # coordinate system, finds the node density per unit volume (3D) 
  # or area (2D) and then converts it to node density per unit area 
  # (3D) or length (2D)                 
  def rho_r(p_r):
    a = np.ones((p_r.shape[0],1))*const_r
    # append constant value needed to rotate back to the right 
    # position                      
    p_r = np.hstack((p_r,a))
    # rotate back to original coordinate system  
    p = np.einsum('ij,...j->...i',R.T,p_r)
    if dim == 2:
      return rho(p)**(1.0/2.0)
    if dim == 3:
      return rho(p)**(2.0/3.0)

  if dim == 2:
    smp_r = np.array([[0],[1]])
    grp_r = np.array([1,2])
    # find the normal vector to the simplex                                                
    v1 = vert[1,:] - vert[0,:]
    normal = rbf.geometry.normal(np.array([v1]))

  if dim == 3:
    smp_r = np.array([[0,1],[0,2],[1,2]])
    grp_r = np.array([1,2,3])
    # find the normal vector to the simplex                                                
    v1 = vert[1,:] - vert[0,:]
    v2 = vert[2,:] - vert[0,:]
    normal = rbf.geometry.normal(np.array([v1,v2]))

  nodes_r,norms_r,groups = rbf.nodegen.volume(
                             rho_r,vert_r,smp_r,
                             groups=grp_r,fix_nodes=fix_r,
                             **kwargs)
  N = nodes_r.shape[0]
  a = np.ones((N,1))*const_r
  nodes_r = np.hstack((nodes_r,a))
  nodes = np.einsum('ij,...j->...i',R.T,nodes_r)

  # pick the normal vector with a positive value for the last 
  # dimension 
  if normal[-1] < 0.0:
    normal *= -1

  normals = np.repeat(normal[None,:],N,axis=0)


  return nodes,normals,groups


def find_free_edges(smp):
  '''                                                                                      
  finds the simplices of all of the unconnected simplex edges 
  '''
  smp = np.asarray(smp,dtype=int)
  dim = smp.shape[1]
  out = []
  sub_smp = []
  # collect all the simplex edges 
  for s in smp:
    for c in combinations(s,dim-1):
      c_list = list(c)
      c_list.sort()
      sub_smp.append(c_list)

  # count the number of repeated edges  
  for s in sub_smp:
    count = 0
    for t in sub_smp:
      if s == t:
        count += 1

    # if the edge appears only once in the collection then it is a
    # free edge              
    if count == 1:
      out.append(s)

  return out


def find_edges(smp):
  '''                                                                                      
  finds the simplices of all of the simplex edges  
  '''
  smp = np.asarray(smp,dtype=int)
  dim = smp.shape[1]
  out = []
  for s in smp:
    for c in combinations(s,dim-1):
      c_list = list(c)
      c_list.sort()
      if c_list not in out:
        out.append(c_list)

  return out


@funtime
def surface(rho,vert,smp,**kwargs):
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  dim = vert.shape[1]

  nodes = np.zeros((0,dim),dtype=float)
  normals = np.zeros((0,dim),dtype=float)
  groups = np.zeros((0),dtype=int)

  # edges of all simplexes on the suface  
  edges = find_edges(smp) # list of lists  

  # free edges of the surface         
  free_edges = find_free_edges(smp) # list of lists  

  edge_nodes = [np.zeros((0,dim))]*len(edges)
  for s in smp:
    fix_nodes = np.zeros((0,dim))
    for e in find_edges([s]):
      fix_nodes = np.vstack((fix_nodes,
                             edge_nodes[edges.index(e)]))
    v = vert[s]
    # there can be more than two simplexes that share the same vertices
    # in 3D and it is necessary to make all vertices into fixed nodes    
    if dim == 3:
      fix_nodes = np.vstack((fix_nodes,v))

    n,m,g1 = nodes_on_simplex(rho,v,fix_nodes=fix_nodes,**kwargs)
    g2 = np.zeros(n.shape[0],dtype=int)
    for i,e in enumerate(find_edges([s])):
      new_edge_nodes = n[g1==i+1]
      edge_nodes[edges.index(e)] = np.vstack((edge_nodes[edges.index(e)],
                                              new_edge_nodes))
      if e in free_edges:
        g2[g1==i+1] = 1

    nodes = np.vstack((nodes,n))
    normals = np.vstack((normals,m))
    groups = np.hstack((groups,g2))

  idx = verify_node_spacing(rho,nodes)
  nodes = nodes[idx]
  normals = normals[idx]
  groups = groups[idx]

  return nodes,normals,groups






