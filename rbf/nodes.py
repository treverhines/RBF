''' 
provides a function for generating a locally quasi-uniform 
distribution of nodes over an arbitrary domain
'''
from __future__ import division
import numpy as np
import rbf.halton
import logging
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from rbf.stencil import stencil_network
from rbf.geometry import (intersection_count,
                          intersection_index,
                          intersection_point,
                          intersection_normal,
                          simplex_outward_normals,
                          contains)
logger = logging.getLogger(__name__)

  
def _default_rho(p):
  '''
  default node density function
  '''
  return np.ones(p.shape[0])


def _neighbors(x,m,p=None,vert=None,smp=None):
  ''' 
  Returns the indices and distances for the `m` nearest neighbors to 
  each node in `x`. If `p` is specified then this function returns the 
  `m` nearest nodes in `p` to each nodes in `x`. Nearest neighbors 
  cannot extend across the boundary defined by `vert` and `smp`.

  Returns
  -------
  (N,m) integer array
    Indices of nearest points.

  (N,m) float array
    Distance to the nearest points.

  '''
  if p is None: 
    p = x
    
  idx = stencil_network(x,p,m,vert=vert,smp=smp)
  dist = np.sqrt(np.sum((x[:,None,:] - p[idx])**2,axis=2))
  return idx,dist


def _neighbor_argsort(nodes,m=None,vert=None,smp=None):
  ''' 
  Returns a permutation array that sorts `nodes` so that each node and 
  its `m` nearest neighbors are close together in memory. This is done 
  through the use of a KD Tree and the Reverse Cuthill-McKee 
  algorithm.

  Returns
  -------
  (N,) int array 
    Sorting indices.

  Examples
  --------
  >>> nodes = np.array([[0.0,1.0],
                        [2.0,1.0],
                        [1.0,1.0]])
  >>> idx = neighbor_argsort(nodes,2)
  >>> nodes[idx]
  array([[ 2.,  1.],
         [ 1.,  1.],
         [ 0.,  1.]])

  '''
  if m is None:
    m = 3**nodes.shape[1]
    
  m = min(m,nodes.shape[0])
  # find the indices of the nearest n nodes for each node
  idx,dist = _neighbors(nodes,m,vert=vert,smp=smp)
  # efficiently form adjacency matrix
  col = idx.ravel()
  row = np.repeat(np.arange(nodes.shape[0]),m)
  data = np.ones(nodes.shape[0]*m,dtype=bool)
  mat = csc_matrix((data,(row,col)),dtype=bool)
  permutation = reverse_cuthill_mckee(mat)
  return permutation


def _snap_to_boundary(nodes,vert,smp,delta=1.0):
  ''' 
  Snaps nodes to the boundary defined by `vert` and `smp`. This is 
  done by slightly shifting each node along the basis directions and 
  checking to see if that caused a boundary intersection. If so, then 
  the intersecting node is reset at the point of intersection. 
  
  Returns
  -------
  (N,D) float array
    New nodes positions.
  
  (N,D) int array
    Index of the simplex that each node is on. If a node is not on a
    simplex (i.e. it is an interior node) then the simplex index is
    -1.

  '''
  n,dim = nodes.shape
  # find the distance to the nearest node
  dx = _neighbors(nodes,2)[1][:,1]
  # allocate output arrays
  out_smpid = np.full(n,-1,dtype=int)
  out_nodes = np.array(nodes,copy=True)
  min_dist = np.full(n,np.inf,dtype=float)
  for i in range(dim):
    for sign in [-1.0,1.0]:
      # `pert_nodes` is `nodes` shifted slightly along dimension `i`
      pert_nodes = np.array(nodes,copy=True)
      pert_nodes[:,i] += delta*sign*dx
      # find which segments intersect the boundary
      idx, = (intersection_count(nodes,pert_nodes,vert,smp) > 0).nonzero()
      # find the intersection points
      pnt = intersection_point(nodes[idx],pert_nodes[idx],vert,smp)
      # find the distance between `nodes` and the intersection point
      dist = np.linalg.norm(nodes[idx] - pnt,axis=1)
      # only snap nodes which have an intersection point that is 
      # closer than any of their previously found intersection points
      snap = dist < min_dist[idx]
      snap_idx = idx[snap]
      out_smpid[snap_idx] = intersection_index(nodes[snap_idx],
                                               pert_nodes[snap_idx],
                                               vert,smp)
      out_nodes[snap_idx] = pnt[snap]
      min_dist[snap_idx] = dist[snap]

  return out_nodes,out_smpid


def _disperse(nodes,rho=None,fixed_nodes=None,m=None,delta=0.1,vert=None,
              smp=None):
  ''' 
  Returns the new position of the free nodes after a dispersal step.
  Nodes on opposite sides of the boundary defined by `vert` and `smp`
  cannot repel eachother. This does not handle node intersections with
  the boundary
  '''
  if rho is None:
    rho = _default_rho
    
  if fixed_nodes is None:
    fixed_nodes = np.zeros((0,nodes.shape[1]),dtype=float)
  
  fixed_nodes = np.asarray(fixed_nodes,dtype=float)  
  if m is None:
    # number of neighbors defaults to 3 raised to the number of 
    # spatial dimensions
    m = 3**nodes.shape[1]

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  m = min(m,nodes.shape[0]+fixed_nodes.shape[0])
  # if m is 0 or 1 then the nodes remain stationary
  if m <= 1:
    return np.array(nodes,copy=True)

  # form collection of all nodes
  all_nodes = np.vstack((nodes,fixed_nodes))
  # find index and distance to nearest nodes
  i,d = _neighbors(nodes,m,p=all_nodes,vert=vert,smp=smp)
  # dont consider a node to be one of its own nearest neighbors
  i,d = i[:,1:],d[:,1:]
  # compute the force proportionality constant between each node
  # based on their charges
  c = 1.0/(rho(all_nodes)[i,None]*rho(nodes)[:,None,None])
  # calculate forces on each node resulting from the `m` nearest nodes 
  forces = c*(nodes[:,None,:] - all_nodes[i,:])/d[:,:,None]**3
  # sum up all the forces for each node
  direction = np.sum(forces,axis=1)
  # normalize the net forces to one
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  
  # move in the direction of the force by an amount proportional to 
  # the distance to the nearest neighbor
  step = delta*d[:,0,None]*direction
  # new node positions
  out = nodes + step
  return out


def _disperse_within_boundary(nodes,vert,smp,rho=None,fixed_nodes=None,
                              m=None,delta=0.1,bound_force=False): 
  '''   
  Returns `nodes` after beingly slightly dispersed within the
  boundaries defined by `vert` and `smp`. The disperson is analogous
  to electrostatic repulsion, where neighboring node exert a repulsive
  force on eachother. If a node is repelled into a boundary then it
  bounces back in.
  '''
  if bound_force:
    bound_vert,bound_smp = vert,smp
  else:
    bound_vert,bound_smp = None,None
  
  # node positions after repulsion 
  out = _disperse(nodes,rho=rho,fixed_nodes=fixed_nodes,m=m,
                  delta=delta,vert=bound_vert,smp=bound_smp)
  # boolean array of nodes which are now outside the domain
  crossed = intersection_count(nodes,out,vert,smp) > 0
  # point where nodes intersected the boundary
  inter = intersection_point(nodes[crossed],out[crossed],vert,smp)
  # normal vector to intersection point
  norms = intersection_normal(nodes[crossed],out[crossed],vert,smp)
  # distance that the node wanted to travel beyond the boundary
  res = out[crossed] - inter
  # bouce node off the boundary
  out[crossed] -= 2*norms*np.sum(res*norms,1)[:,None]        
  # check to see if the bounced nodes still intersect the boundary. If 
  # not then set the bounced nodes back to their original position
  crossed = intersection_count(nodes,out,vert,smp) > 0
  out[crossed] = nodes[crossed]
  return out
  

def _form_node_groups(smpid,boundary_groups=None):
  '''
  Create a dictionary identifying the nodes in each group. 
  '''
  if boundary_groups is None:
    # by default there is one boundary group containing all the
    # boundary nodes
    boundary_groups = {'boundary':range(smpid.shape[0])}

  if 'interior' in boundary_groups:
    raise ValueError('"interior" is a reserved group name')
  
  # put the nodes into groups based on which (if any) simplex the
  # nodes are attached to. The groups are defined in the
  # `boundary_groups` dictionary.
  indices = {}
  # Interior nodes have a smpid of -1
  indices['interior'] = [i for i,s in enumerate(smpid) if s == -1]
  indices['interior'] = np.array(indices['interior'],dtype=int)
  # Form the boundary groups
  for group_name,group_smp in boundary_groups.items():
    indices[group_name] = [i for i,s in enumerate(smpid) if s in group_smp]
    indices[group_name] = np.array(indices[group_name],dtype=int)

  # See if each node belongs to a group. print a warning if not
  unique_indices = np.unique(np.hstack(v for v in indices.values()))
  if unique_indices.shape[0] != smpid.shape[0]:
    logger.warning('not all nodes belong to a group')
  
  return indices  


def _sort_nodes(nodes,indices):
  '''
  sort so that nodes that are close in space are also close in memory
  '''
  sort_idx = _neighbor_argsort(nodes)
  out_nodes = nodes[sort_idx]
  # update the indices because of this sorting
  out_indices = indices.copy()
  reverse_sort_idx = np.argsort(sort_idx)
  for key,val in indices.items():
    out_indices[key] = reverse_sort_idx[val]
  
  return out_nodes, out_indices  


def _form_normal_vectors(smpid,vert,smp,indices):
  '''
  Form the normal vectors for each node group except for the interior
  nodes
  '''
  normals = {} 
  # get the normal vectors for each simplex  
  simplex_normals = simplex_outward_normals(vert,smp)
  for group_name in indices.keys():
    # dont form normal vectors for the interior nodes
    if group_name == 'interior':
      continue
    
    idx = indices[group_name]
    # make sure that we are creating normal vectors only for boundary
    # nodes
    if np.any(smpid[idx] == -1):
      raise ValueError('cannot create normal vectors for interior '
                       'nodes')

    # get the normal vectors for each node in this group
    normals[group_name] = simplex_normals[smpid[idx]]

  return normals

  
def _form_ghost_nodes(nodes,smpid,vert,smp,indices,
                      boundary_groups_with_ghosts):      
  '''
  add ghost nodes to the node set
  '''
  out_nodes = nodes.copy()
  out_smpid = smpid.copy()
  out_indices = indices.copy()

  # get the normal vectors for each simplex  
  simplex_normals = simplex_outward_normals(vert,smp)

  # get the shortest distance between any two nodes. This will be used
  # to determine how far the ghost nodes should be from the boundary
  dx = np.min(_neighbors(nodes,2)[1][:,1])

  for group_name in boundary_groups_with_ghosts:
    # get the indices of nodes in this group
    idx = indices[group_name]
    # make sure that we are creating ghost nodes only for boundary nodes
    if np.any(smpid[idx] == -1):
      raise ValueError('cannot create ghost nodes for interior nodes')

    # get the normal vectors for these nodes
    normals = simplex_normals[smpid[idx]]
    # create ghost nodes for this group
    ghosts = nodes[idx] + dx*normals

    # append the ghost nodes to the output
    out_nodes = np.vstack((out_nodes,ghosts))
    # record the simplex that each ghost node is associated with
    out_smpid = np.hstack((out_smpid,smpid[idx]))
    # record the ghost node indices for this group
    start = out_nodes.shape[0] - ghosts.shape[0] 
    stop = out_nodes.shape[0] 
    out_indices[group_name + '_ghosts'] = np.arange(start,stop)

  return out_nodes, out_smpid, out_indices
  

def _rejection_sampling_nodes(N, vert, smp, rho=None, max_sample_size=1000000):
  '''
  Returns `N` nodes within the boundaries defined by `vert` and `smp`
  and with density `rho`. The nodes are generated by rejection
  sampling.

  Parameters
  ----------
  nodes : (N,D) float array

  vert : (P,D) float array

  smp : (Q,D) int array
    
  rho : function

  max_sample_size : int
    max number of nodes allowed in a sample for the rejection
    algorithm. This prevents excessive RAM usage
     
  '''
  if rho is None:
    rho = _default_rho

  # form bounding box for the domain so that a RNG can produce values
  # that mostly lie within the domain
  lb = np.min(vert,axis=0)
  ub = np.max(vert,axis=0)
  ndim = vert.shape[1]
  # form Halton sequence generator
  H = rbf.halton.Halton(ndim+1)
  # initiate array of nodes
  nodes = np.zeros((0,ndim),dtype=float)
  # node counter
  total_samples = 0
  # I use a rejection algorithm to get a sampling of nodes that
  # resemble to density specified by rho. The acceptance keeps track
  # of the ratio of accepted nodes to tested nodes
  acceptance = 1.0
  while nodes.shape[0] < N:
    # to keep most of this loop in cython and c code, the rejection
    # algorithm is done in chunks.  The number of samples in each 
    # chunk is a rough estimate of the number of samples needed in
    # order to get the desired number of accepted nodes.
    if acceptance == 0.0:
      sample_size = max_sample_size    
    else:
      # estimated number of samples needed to get N accepted nodes
      sample_size = int(np.ceil((N-nodes.shape[0])/acceptance))
      # dont let sample_size exceed max_sample_size
      sample_size = min(sample_size,max_sample_size)

    # In order for a test node to be accepted, rho evaluated at that
    # test node needs to be larger than a random number with uniform
    # distribution between 0 and 1. Here I form the test nodes and
    # those random numbers
    seq = H(sample_size)
    test_nodes,seq1d = seq[:,:-1],seq[:,-1]
    # scale the range of test node to encompass the domain  
    test_nodes = (ub-lb)*test_nodes + lb
    # reject test points based on random value
    test_nodes = test_nodes[rho(test_nodes) > seq1d]
    # reject test points that are outside of the domain
    test_nodes = test_nodes[contains(test_nodes,vert,smp)]
    # append what remains to the collection of accepted nodes. If
    # there are too many new nodes, then cut it back down so the total
    # size is `N`
    if (test_nodes.shape[0] + nodes.shape[0]) > N:
      test_nodes = test_nodes[:(N - nodes.shape[0])]
      
    nodes = np.vstack((nodes,test_nodes))
    logger.debug('accepted %s of %s nodes' % (nodes.shape[0],N))
    # update the acceptance. the acceptance is the ratio of accepted
    # nodes to sampled nodes
    total_samples += sample_size
    acceptance = nodes.shape[0]/total_samples

  return nodes
  

def min_energy_nodes(N,vert,smp,
                     rho=None,
                     fixed_nodes=None,
                     itr=100,
                     m=None,
                     delta=0.05,
                     boundary_groups=None,
                     boundary_groups_with_ghosts=None,
                     bound_force=False):
  ''' 
  Generates nodes within a 1, 2, or 3 dimensional domain using a 
  minimum energy algorithm.
  
  The algorithm is as follows. A random distribution of nodes is first 
  generated within the domain. The nodes positions are then 
  iteratively adjusted. For each iteration, the nearest neighbors to 
  each node are found. A repulsion force is calculated for each node 
  using the distance to its nearest neighbors and their charges (which 
  are inversely proportional to the node density). Each node then 
  moves in the direction of the net force acting on it. 

  Parameters
  ----------
  N : int
    Number of nodes.
      
  vert : (P,D) array
    Vertices making up the boundary.

  smp : (Q,D) array
    Describes how the vertices are connected to form the boundary.
    
  rho : function, optional
    Node density function. Takes a (N,D) array of coordinates in D 
    dimensional space and returns an (N,) array of densities which 
    have been normalized so that the maximum density in the domain 
    is 1.0. This function will still work if the maximum value is 
    normalized to something less than 1.0; however it will be less 
    efficient.

  fixed_nodes : (F,D) array, optional
    Nodes which do not move and only provide a repulsion force.
 
  itr : int, optional
    Number of repulsion iterations. If this number is small then the 
    nodes will not reach a minimum energy equilibrium.

  m : int, optional
    Number of neighboring nodes to use when calculating the repulsion 
    force. When `m` is small, the equilibrium state tends to be a 
    uniform node distribution (regardless of `rho`), when `m` is 
    large, nodes tend to get pushed up against the boundaries.

  delta : float, optional
    Scaling factor for the node step size in each iteration. The 
    step size is equal to `delta` times the distance to the nearest 
    neighbor.

  boundary_groups: dict, optional
    Dictionary defining the boundary groups. The keys are the names of
    the groups and the values are lists of simplex indices making up
    each group. This defaults to one group named 'boundary' which is
    made up of all the simplices.

  boundary_groups_with_ghosts: list of strs, optional
    List of boundary groups that will be given ghost nodes. By
    default, no groups are given ghost nodes
      
  bound_force : bool, optional
    If True, then nodes cannot repel other nodes through the domain 
    boundary. Set to True if the domain has edges that nearly touch 
    eachother. Setting this to True may significantly increase 
    computation time.

  Returns
  -------
  nodes: (N,D) float array 
    Nodes positions.

  smpid: (N,) int array
    Index of the simplex that each node is on. If a node is not on a 
    simplex (i.e. it is an interior node) then the simplex index is 
    -1.

  Notes
  -----
  It is assumed that `vert` and `smp` define a closed domain. If 
  this is not the case, then it is likely that an error message will 
  be raised which says "ValueError: No intersection found for 
  segment ...".
      
  '''
  logger.debug('starting minimum energy node generation') 
  vert = np.asarray(vert,dtype=float) 
  smp = np.asarray(smp,dtype=int) 

  logger.debug('finding node positions with rejection sampling')
  nodes = _rejection_sampling_nodes(N,vert,smp,rho=rho)

  # use a minimum energy algorithm to spread out the nodes
  for i in range(itr):
    logger.debug('starting node repulsion iteration %s of %s' % (i+1,itr)) 
    nodes = _disperse_within_boundary(nodes,vert,smp,rho=rho,
                                      fixed_nodes=fixed_nodes,m=m,
                                      delta=delta,bound_force=bound_force)

  logger.debug('snapping nodes to boundary') 
  nodes,smpid = _snap_to_boundary(nodes,vert,smp,delta=0.5)

  # get the indices of nodes belonging to each group
  logger.debug('assigning nodes to groups') 
  indices = _form_node_groups(smpid,boundary_groups) 

  # form ghost nodes for the specified groups
  if boundary_groups_with_ghosts:
    logger.debug('creating ghost nodes for groups: %s' % 
                 ', '.join(boundary_groups_with_ghosts))
    nodes,smpid,indices = _form_ghost_nodes(
                            nodes,smpid,vert,smp,indices,
                            boundary_groups_with_ghosts)

  # form the normal vectors for the boundary groups
  logger.debug('creating normal vectors for boundary nodes')
  normals = _form_normal_vectors(smpid,vert,smp,indices)
  
  # sort `nodes` so that spatially adjacent nodes are close together
  # in memory. Update `indices` so that it is still pointing to the
  # same nodes
  logger.debug('sorting nodes so that neighboring nodes are close in '
               'memory') 
  nodes,indices = _sort_nodes(nodes,indices)

  logger.debug('finished generating %s nodes' % nodes.shape[0])
  return nodes, indices, normals



