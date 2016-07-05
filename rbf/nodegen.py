#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.halton
import rbf.geometry as gm
import rbf.integrate
import rbf.stencil
import logging
import scipy.sparse
from itertools import combinations
logger = logging.getLogger(__name__)

def adjacency_argsort(nodes,n=10):
  ''' 
  Description
  -----------
    Sorts nodes so that adjacent nodes are close together. This is done
    primarily through use of a KD Tree and the Reverse Cuthill-McKee 
    algorithm

  Parameters
  ----------
    nodes: (N,D) array of nodes

    n: number of adjacencies to identify for each node. The           
       permutation array will place adjacent nodes close to eachother
       in memory.  This should be about equal to the stencil size for 
       RBF-FD method.

  Returns
  -------
    permutation: (N,) array of sorting indices
  '''
  nodes = np.asarray(nodes,dtype=float)
  n = min(n,nodes.shape[0])

  # find the indices of the nearest N nodes for each node
  idx,dist = rbf.stencil.nearest(nodes,nodes,n)

  # efficiently form adjacency matrix
  col = idx.flatten()
  row = np.repeat(np.arange(nodes.shape[0]),n)
  data = np.ones(nodes.shape[0]*n,dtype=bool)
  M = scipy.sparse.csr_matrix((data,(row,col)),dtype=bool)
  permutation = scipy.sparse.csgraph.reverse_cuthill_mckee(M)

  return permutation

def verify_node_spacing(rho,nodes,tol=0.25):
  ''' 
  Description
  -----------
    Returns indices of nodes which are consistent with the node
    density function rho. Indexing nodes with the output will return a
    pared down node set that is consistent with rho

  Parameters
  ----------
    rho: callable node density function
  
    nodes: (N,D) array of nodes

    tol: if a nodes nearest neighbor deviates by more than this factor
      from the expected node density, then the node is removed

  Returns 
  ------- 
    array of indices which are consistent with the node density
    function

  '''
  logger.info('verifying node spacing')  
  dim = nodes.shape[1]
  keep_indices = np.arange(nodes.shape[0],dtype=int)
  # minimum distance allowed between nodes
  mindist = tol/(rho(nodes)**(1.0/dim))
  s,dx = rbf.stencil.nearest(nodes,nodes,2)
  while np.any(dx[:,1] < mindist):
    toss = np.argmin(dx[:,1]/mindist)
    logger.info('removing node %s for being too close to adjacent node' 
                % keep_indices[toss])
    nodes = np.delete(nodes,toss,axis=0)
    keep_indices = np.delete(keep_indices,toss,axis=0)
    mindist = tol/(rho(nodes)**(1.0/dim))
    s,dx = rbf.stencil.nearest(nodes,nodes,2)

  return keep_indices


def merge_nodes(**kwargs):
  out_dict = {}
  out_array = ()
  n = 0
  for k,a in kwargs.items():
    a = np.array(a,dtype=np.float64,order='c',copy=True)
    idx = range(n,n+a.shape[0])
    n += a.shape[0]
    out_array += a,
    out_dict[k] = idx

  out_array = np.vstack(out_array)
  return out_array,out_dict


def _repel_step(free_nodes,rho,
                fix_nodes=None,
                n=10,delta=0.1):
  ''' 
  returns the new position of the free nodes after a repulsion step
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  # if n is 0 or 1 then the nodes remain stationary
  if n <= 1:
    return free_nodes

  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)

  # form collection of all nodes
  nodes = np.vstack((free_nodes,fix_nodes))

  # find index and distance to nearest nodes
  i,d = rbf.stencil.nearest(free_nodes,nodes,n)

  # dont consider a node to be one of its own nearest neighbors
  i = i[:,1:]
  d = d[:,1:]

  # compute the force proportionality constant between each node
  # based on their charges
  c = 1.0/(rho(nodes)[i,None]*rho(free_nodes)[:,None,None])

  # sum up all the forces on each node
  direction = np.sum(c*(free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)

  # normalize the forces to one
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  

  # move in the direction of the force by an amount proportional to 
  # the distance to the nearest neighbor
  step = delta*d[:,0,None]*direction

  # new node positions
  free_nodes += step
  return free_nodes


def _repel_bounce(free_nodes,vertices,
                  simplices,rho,   
                  fix_nodes=None,
                  itr=20,n=10,delta=0.1,
                  max_bounces=3):
  ''' 
  nodes are repelled by eachother and bounce off boundaries
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 
  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)

  # this is used for the lengthscale of the domain
  scale = np.max(vertices) - np.min(vertices)

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    # node positions after repulsion 
    free_nodes_new = _repel_step(free_nodes,rho,
                                 fix_nodes=fix_nodes,
                                 n=n,delta=delta)

    # boolean array of nodes which are now outside the domain
    crossed = ~gm.contains(free_nodes_new,vertices,simplices)
    bounces = 0
    while np.any(crossed):
      # point where nodes intersected the boundary
      inter = gm.intersection_point(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vertices,simplices)

      # normal vector to intersection point
      norms = gm.intersection_normal(
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
        crossed = ~gm.contains(free_nodes_new,vertices,simplices)
        bounces += 1

    free_nodes = free_nodes_new  
  
  return free_nodes


def _repel_stick(free_nodes,vertices,
                 simplices,rho,   
                 fix_nodes=None,
                 itr=20,n=10,delta=0.1,
                 max_bounces=3):
  ''' 
  nodes are repelled by eachother and then become fixed when they hit 
  a boundary
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)
  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 

  if fix_nodes is None:
    fix_nodes = np.zeros((0,free_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes,dtype=float)

  # this array will be populated 
  node_group = np.repeat(-1,free_nodes.shape[0])

  # length scale of the domain
  scale = np.max(vertices) - np.min(vertices)

  # ensure that the number of nodes used to compute repulsion force is
  # less than or equal to the total number of nodes
  if n > (free_nodes.shape[0]+fix_nodes.shape[0]):
    n = free_nodes.shape[0]+fix_nodes.shape[0]

  for k in range(itr):
    # indices of all nodes not associated with a simplex (i.e. free 
    # nodes)
    ungrouped = np.nonzero(node_group==-1)[0]

    # indices of nodes associated with a simplex (i.e. nodes which 
    # intersected a boundary
    grouped = np.nonzero(node_group>=0)[0]

    # nodes which are stationary 
    grouped_free_nodes = free_nodes[grouped]    
    all_fix_nodes = np.vstack((fix_nodes,grouped_free_nodes))

    # nodes which are free
    ungrouped_free_nodes = free_nodes[ungrouped]

    # new position of free nodes
    ungrouped_free_nodes_new = _repel_step(
                                 ungrouped_free_nodes,rho,
                                 fix_nodes=all_fix_nodes,
                                 n=n,delta=delta)

    # indices of free nodes which crossed a boundary
    crossed = ~gm.contains(ungrouped_free_nodes_new,vertices,simplices)
  
    # if a node intersected a boundary then associate it with a simplex
    node_group[ungrouped[crossed]] = gm.intersection_index(
                                       ungrouped_free_nodes[crossed],     
                                       ungrouped_free_nodes_new[crossed], 
                                       vertices,simplices)

    # outward normal vector at intesection points
    norms = gm.intersection_normal(
              ungrouped_free_nodes[crossed],     
              ungrouped_free_nodes_new[crossed], 
              vertices,simplices)

    # intersection point for nodes which crossed a boundary
    inter = gm.intersection_point(
              ungrouped_free_nodes[crossed],     
              ungrouped_free_nodes_new[crossed],
              vertices,simplices)

    # new position of nodes which crossed the boundary is just within
    # the intersection point
    ungrouped_free_nodes_new[crossed] = inter - 1e-10*scale*norms
    free_nodes[ungrouped] = ungrouped_free_nodes_new

  return free_nodes,node_group

def volume(rho,vertices,simplices,fix_nodes=None,
           itr=20,n=10,delta=0.1,check_simplices=True,
           sort_nodes=True):
  ''' 
  Generates nodes within the D-dimensional volume enclosed by the 
  simplexes using a minimum energy algorithm.  At each iteration 
  the nearest neighbors to each node are found and then a repulsion
  force is calculated using the distance to the nearest neighbors and
  their charges (which is inversely proportional to the node density).
  Each node then moves in the direction of the net force acting on it.  
  The step size is equal to delta times the distance to the nearest 
  node.  

  Paramters
  ---------
    rho: node density function. Takes a (N,D) array of coordinates
      in D dimensional space and returns an (N,) array of node
      densities at those coordinates.  Can also specify an integer 
      number of nodes if the density is to be uniform

    vertices: boundary vertices

    simplices: describes how the vertices are connected to form the 
      boundary
    
    fix_nodes (default=None): Nodes which do not move and only provide
      a repulsion force
 
    itr (default=20): number of repulsion iterations.  If this number
      is small then the nodes will not reach a minimum energy
      equilibrium.

    n (default=10): number of neighboring nodes to use when calculating
      repulsion force. When n is small, the equilibrium state tends to
      be a uniform node distribution (regardless of the specified
      rho), when n is large, nodes tend to get pushed up against the
      boundaries.  It is best to use a small n and a value for itr
      which is large enough for the nodes to disperse a little bit but
      not reach equilibrium.

    delta (default=0.1): Controls the node step size for each
      iteration.  The step size is equal to delta times the distance to
      the nearest neighbor

    check_simplices (default=True): Identifies whether the simplices 
      should be sorted so that their normal vectors point outward. 
      This only matters if 'rho' is a scalar, in which case the 
      volume/area of the domain must be calculated. Calculating the 
      volume/area requires properly oriented simplices

    sort_nodes (default=True): If True, nodes that are close in space
      will also be close in memory. This is done with the Reverse 
      Cuthill-McKee algorithm


  Returns
  -------
    nodes: (N,D) array of nodes 

    simplex_indices: (N,) Index of the simplex that each node is on.
      If a node is not on a simplex (i.e. it is an interior node) then
      the simplex index is -1.

  '''
  max_sample_size = 1000000

  vertices = np.asarray(vertices,dtype=float) 
  simplices = np.asarray(simplices,dtype=int) 


  # if rho is a scalar rather than a density function then compute the
  # volume of the simplicial complex and create a rho function with
  # uniform density that integrates to the specified scalar
  if np.isscalar(rho):
    if check_simplices:
      simplices = gm.oriented_simplices(vertices,simplices)

    N = int(np.round(rho))
    vol = rbf.geometry.enclosure(vertices,simplices,orient=False)
    if (vol < 0.0):
      raise ValueError(
        'simplicial complex found to have a negative volume. Check the '
        'orientation of simplices and ensure closedness')
   
    err = 0.0
    minval = N/vol
    maxval = N/vol
    def rho(p):
      return np.repeat(N/vol,p.shape[0])

  # if rho is a callable function then integrate it to find the total
  # number of nodes
  else:
    N,err,minval,maxval = rbf.integrate.rmcint(rho,vertices,simplices)

  assert minval >= 0.0, (
    'values in node density function must be positive')
  
  # total number of nodes 
  N = int(np.round(N))

  # node density function normalized to 1
  def rho_normalized(p):
    return rho(p)/maxval

  # form bounding box for the domain so that a RNG can produce values
  # that mostly lie within the domain
  lb = np.min(vertices,0)
  ub = np.max(vertices,0)

  ndim = lb.shape[0]
  # form Halton number generator
  H = rbf.halton.Halton(ndim+1)

  # initiate array of nodes
  nodes = np.zeros((0,ndim))

  # node counter
  cnt = 0

  # I use a rejection algorithm to get an initial sampling of nodes 
  # that resemble to density specified by rho. The acceptance keeps
  # track of the ratio of accepted nodes to tested nodes
  acceptance = 1.0

  while nodes.shape[0] < N:
    # to keep most of this loop in cython and c code, the rejection
    # algorithm is done in chunks.  The number of samples in each 
    # chunk is a rough estimate of the number of samples needed in
    # order to get the desired number of accepted nodes.
    if acceptance == 0.0:
      sample_size = max_sample_size    
    else:
      sample_size = int(np.ceil((N-nodes.shape[0])/acceptance))
      if sample_size > max_sample_size:
        sample_size = max_sample_size

    cnt += sample_size
    # form test points
    seqNd = H(sample_size)

    # In order for a test point to be accepted, rho evaluated at that 
    # test point needs to be larger than a random number with uniform 
    # distribution between 0 and 1. Here I form those random numbers
    seq1d = seqNd[:,-1]

    # scale range of test points to encompass the domain  
    new_nodes = (ub-lb)*seqNd[:,:ndim] + lb

    # reject test points based on random value
    new_nodes = new_nodes[rho_normalized(new_nodes) > seq1d]

    # reject test points that are outside of the domain
    new_nodes = new_nodes[gm.contains(new_nodes,vertices,simplices)]

    # append to collection of accepted nodes
    nodes = np.vstack((nodes,new_nodes))

    logger.info('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]

  # use a minimum energy algorithm to spread out the nodes
  logger.info('repelling nodes with boundary bouncing') 
  nodes = _repel_bounce(nodes,vertices,simplices,rho_normalized,
                        fix_nodes=fix_nodes,itr=itr,n=n,delta=delta)

  logger.info('repelling nodes with boundary sticking') 
  nodes,smpid = _repel_stick(nodes,vertices,simplices,rho_normalized,
                             fix_nodes=fix_nodes,itr=itr,n=n,delta=delta)

  # make sure nodes are sufficienty far away
  idx = verify_node_spacing(rho,nodes)
  nodes = nodes[idx]
  smpid = smpid[idx]

  # sort so that nodes that are close in space are also close in memory
  if sort_nodes:
    idx = adjacency_argsort(nodes,n=n)
    nodes = nodes[idx]
    smpid = smpid[idx] 
  
  return nodes,smpid


def _simplex_rotation(vert):
  '''                                                                                      
  returns a matrix that rotates the simplex such that      
  its normal is pointing in the direction of the last axis    
  '''
  vert = np.asarray(vert)
  dim = vert.shape[1]
  if dim == 2:
    # anchor one node of the simplex at the origin      
    normal = rbf.geometry.simplex_normals(vert,[[0,1]])[0]

    # find the angle between the y axis and the simplex normal   
    argz = np.arctan2(normal[0],normal[1])

    # create a rotation matrix that rotates the normal onto the y axis
    R = np.array([[np.cos(argz), -np.sin(argz)],
                  [np.sin(argz),  np.cos(argz)]])
    return R

  if dim == 3:
    # find the normal vector to the simplex      
    normal = rbf.geometry.simplex_normals(vert,[[0,1,2]])[0]

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


def _nodes_on_simplex(rho,vert,fix_nodes=None,**kwargs):
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
  R = _simplex_rotation(vert)

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

  if dim == 3:
    smp_r = np.array([[0,1],[0,2],[1,2]])

  nodes_r,smpid = rbf.nodegen.volume(
                    rho_r,vert_r,smp_r,
                    fix_nodes=fix_r,
                    **kwargs)
  N = nodes_r.shape[0]
  a = np.ones((N,1))*const_r
  nodes_r = np.hstack((nodes_r,a))
  nodes = np.einsum('ij,...j->...i',R.T,nodes_r)

  return nodes,smpid


def _find_free_edges(smp):
  '''                                                                                      
  finds the vertex indices making up all of the unconnected simplex edges 
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


def _find_edges(smp):
  '''                                                                                      
  finds the vertex indices making up all of the simplex edges  
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


def surface(rho,vert,smp,**kwargs):
  ''' 
  Description
  -----------
    returns nodes in N-D space which lie on a hypersurface
    defined by the given simplexes

  Parameters
  ----------
    rho: Node density function for N-D space. Note, this is NOT the
      node density on the hyperplane

    vert: simplex vertices 
  
    smp: vertex indices for each simplex

  Returns
  -------
    nodes: nodes on the surface

    smpid: index of the simplex that each node is one

    is_boundary: boolean array identifying whether the node exists on the 
      edge of the surface

  '''    
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  dim = vert.shape[1]

  nodes = np.zeros((0,dim),dtype=float)
  groups = np.zeros((0),dtype=int)
  is_boundary = np.zeros((0),dtype=bool)

  # edges of all simplexes on the suface  
  edges = _find_edges(smp) # list of lists  

  # free edges of the surface         
  free_edges = _find_free_edges(smp) # list of lists  

  # keep track of all nodes which have stuck to a simplex edge
  edge_nodes = [np.zeros((0,dim))]*len(edges)
  for sidx,s in enumerate(smp):
    fix_nodes = np.zeros((0,dim))
    # find any existing nodes which are on this simplex's edges
    # and consider them to be fixed nodes
    for e in _find_edges([s]):
      fix_nodes = np.vstack((fix_nodes,
                             edge_nodes[edges.index(e)]))

    # there can be more than two simplexes that share the same vertices
    # in 3D and it is necessary to make all vertices into fixed nodes.
    # The final node set will not contain any of the surface vertices
    # (this may later change)
    if dim == 3:
      fix_nodes = np.vstack((fix_nodes,vert[s]))

    n,g = _nodes_on_simplex(rho,vert[s],fix_nodes=fix_nodes,**kwargs)

    is_boundary_i = np.zeros(n.shape[0],dtype=bool)
    for i,e in enumerate(_find_edges([s])):
      new_edge_nodes = n[g==i]
      edge_nodes[edges.index(e)] = np.vstack((edge_nodes[edges.index(e)],
                                              new_edge_nodes))
      # if the current edge is a free edge then assign its simplex index
      # to be sidx 
      if e in free_edges:
        is_boundary_i[g==i] = True

    nodes = np.vstack((nodes,n))
    groups = np.hstack((groups,np.repeat(sidx,n.shape[0])))
    is_boundary = np.hstack((is_boundary,is_boundary_i))

  idx = verify_node_spacing(rho,nodes)
  nodes = nodes[idx]
  groups = groups[idx]
  is_boundary = is_boundary[idx] 

  return nodes,groups,is_boundary






