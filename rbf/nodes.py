''' 
provides a function for generating a locally quasi-uniform 
distribution of nodes over an arbitrary domain
'''
from __future__ import division
import numpy as np
import rbf.halton
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from rbf.stencil import stencil_network
from rbf.geometry import (intersection_count,
                          intersection_index,
                          intersection_point,
                          intersection_normal,
                          contains)
logger = logging.getLogger(__name__)


def neighbors(x,m,p=None,vert=None,smp=None):
  ''' 
  Returns the indices and distances for the *m* nearest neighbors to 
  each node in *x*. If *p* is specified then this function returns the 
  *m* nearest nodes in *p* to each nodes in *x*. Nearest neighbors 
  cannot extend across the boundary defined by *vert* and *smp*.
  
  Parameters
  ----------
  x : (N,D) array
    Node positions.
    
  m : integer
    Number of neighbors to find for each point in *x*.
    
  p : (M,D) array, optional
    Node positions.
        
  vert : (P,D) array, optional     
    Vertices of the boundary.

  smp : (Q,D) array, optional  
    Connectivity of vertices to form the boundary.
    
  Returns
  -------
  idx : (N,*m*) integer array
    Indices of nearest points.

  dist : (N,*m*) float array
    Distance to the nearest points.

  '''
  x = np.asarray(x,dtype=float)
  if p is None: p = x
  idx = stencil_network(x,p,m,vert=vert,smp=smp)
  dist = np.sqrt(np.sum((x[:,None,:] - p[idx])**2,axis=2))
  return idx,dist


def neighbor_argsort(nodes,m=10,vert=None,smp=None):
  ''' 
  Returns a permutation array that sorts *nodes* so that each node and 
  its *m* nearest neighbors are close together in memory. This is done 
  through the use of a KD Tree and the Reverse Cuthill-McKee 
  algorithm.

  Parameters
  ----------
  nodes : (N,D) array
    Node positions.

  m : int, optional
    Number of neighboring nodes to place close together in memory. 
    This should be about equal to the stencil size for RBF-FD method.

  Returns
  -------
  permutation: (N,) array 
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
  nodes = np.asarray(nodes,dtype=float)
  m = min(m,nodes.shape[0])
  # find the indices of the nearest n nodes for each node
  idx,dist = neighbors(nodes,m,vert=vert,smp=smp)
  # efficiently form adjacency matrix
  col = idx.ravel()
  row = np.repeat(np.arange(nodes.shape[0]),m)
  data = np.ones(nodes.shape[0]*m,dtype=bool)
  mat = csr_matrix((data,(row,col)),dtype=bool)
  permutation = reverse_cuthill_mckee(mat)
  return permutation


def snap_to_boundary(nodes,vert,smp,delta=1.0):
  ''' 
  Snaps nodes to the boundary defined by *vert* and *smp*. This is 
  done by slightly shifting each node along the basis directions and 
  checking to see if that caused a boundary intersection. If so, then 
  the intersecting node is reset at the point of intersection.

  Parameters
  ----------
  nodes : (N,D) float array
    Node positions.
  
  vert : (M,D) float array
    Vertices making up the boundary.
  
  smp : (P,D) int array
    Connectivity of the vertices to form the boundary. Each row 
    contains the indices of the vertices which form one simplex of the 
    boundary.
    
  delta : float, optional
    Controls the maximum snapping distance. The maximum snapping 
    distance for each node is *delta* times the distance to the 
    nearest neighbor. This defaults to 1.0.
  
  Returns
  -------
  out_nodes : (N,D) float array
    New nodes positions.
  
  out_smp : (N,D) int array
    Index of the simplex that each node is on. If a node is not on a 
    simplex (i.e. it is an interior node) then the simplex index is 
    -1.
    
  '''
  nodes = np.asarray(nodes,dtype=float)
  vert = np.asarray(vert,dtype=float)
  smp = np.asarray(smp,dtype=int)
  n,dim = nodes.shape
  # find the distance to the nearest node
  dx = neighbors(nodes,2)[1][:,1]
  # allocate output arrays
  out_smpid = np.full(n,-1,dtype=int)
  out_nodes = np.array(nodes,copy=True)
  min_dist = np.full(n,np.inf,dtype=float)
  for i in range(dim):
    for sign in [-1.0,1.0]:
      # *pert_nodes* is *nodes* shifted slightly along dimension *i*
      pert_nodes = np.array(nodes,copy=True)
      pert_nodes[:,i] += delta*sign*dx
      # find which segments intersect the boundary
      idx, = (intersection_count(nodes,pert_nodes,vert,smp) > 0).nonzero()
      # find the intersection points
      pnt = intersection_point(nodes[idx],pert_nodes[idx],vert,smp)
      # find the distance between *nodes* and the intersection point
      dist = np.linalg.norm(nodes[idx] - pnt,axis=1)
      # only snap nodes which have an intersection point that is 
      # closer than any of their previously found intersection points
      snap = dist < min_dist[idx]
      snap_idx = idx[snap]
      out_smpid[snap_idx] = intersection_index(nodes[snap_idx],pert_nodes[snap_idx],vert,smp)
      out_nodes[snap_idx] = pnt[snap]
      min_dist[snap_idx] = dist[snap]

  return out_nodes,out_smpid


def _disperse(free_nodes,fix_nodes,rho,m,delta,vert,smp):
  ''' 
  Returns the new position of the free nodes after a dispersal step. 
  Nodes on opposite sides of the boundary defined by vert and smp 
  cannot repel eachother.
  '''
  # if m is 0 or 1 then the nodes remain stationary
  if m <= 1:
    return np.array(free_nodes,copy=True)

  # form collection of all nodes
  nodes = np.vstack((free_nodes,fix_nodes))
  # find index and distance to nearest nodes
  i,d = neighbors(free_nodes,m,p=nodes,vert=vert,smp=smp)
  # dont consider a node to be one of its own nearest neighbors
  i,d = i[:,1:],d[:,1:]
  # compute the force proportionality constant between each node
  # based on their charges
  c = 1.0/(rho(nodes)[i,None]*rho(free_nodes)[:,None,None])
  # calculate forces on each node resulting from the *n* nearest nodes 
  forces = c*(free_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3
  # sum up all the forces for each node
  direction = np.sum(forces,1)
  # normalize the net forces to one
  direction /= np.linalg.norm(direction,axis=1)[:,None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)  
  # move in the direction of the force by an amount proportional to 
  # the distance to the nearest neighbor
  step = delta*d[:,0,None]*direction
  # new node positions
  out = free_nodes + step
  return out


def disperse(nodes,vert=None,smp=None,rho=None,fix_nodes=None,
             m=None,delta=0.1,bound_force=False): 
  '''   
  Returns *nodes* after beingly slightly dispersed. The disperson is 
  analogous to electrostatic repulsion, where neighboring node exert a 
  repulsive force on eachother. The repulsive force for each node is 
  constant, by default, but it can vary spatially by specifying *rho*. 
  If a node intersects the boundary defined by *vert* and *smp* then 
  it will bounce off the boundary elastically. This ensures that no 
  nodes will leave the domain, assuming the domain is closed and all 
  nodes are initially inside. Using the electrostatic analogy, this 
  function returns the nodes after a single *time step*, and greater 
  amounts of dispersion can be attained by calling this function 
  iteratively.
  
  Parameters
  ----------
  nodes : (N,D) float array
    Node positions.

  vert : (P,D) array, optional
    Boundary vertices.

  smp : (Q,D) array, optional
    Describes how the vertices are connected to form the boundary. 
    
  rho : function, optional
    Node density function. Takes a (N,D) array of coordinates in D 
    dimensional space and returns an (N,) array of densities which 
    have been normalized so that the maximum density in the domain 
    is 1.0. This function will still work if the maximum value is 
    normalized to something less than 1.0; however it will be less 
    efficient.

  fix_nodes : (F,D) array, optional
    Nodes which do not move and only provide a repulsion force.

  m : int, optional
    Number of neighboring nodes to use when calculating the repulsion 
    force. When *m* is small, the equilibrium state tends to be a 
    uniform node distribution (regardless of *rho*), when *m* is 
    large, nodes tend to get pushed up against the boundaries.

  delta : float, optional
    Scaling factor for the node step size. The step size is equal to 
    *delta* times the distance to the nearest neighbor.

  bound_force : bool, optional
    If True, then nodes cannot repel other nodes through the domain 
    boundary. Set to True if the domain has edges that nearly touch 
    eachother. Setting this to True may significantly increase 
    computation time.
    
  Returns
  -------
  out : (N,D) float array
    Nodes after being dispersed.
    
  '''
  nodes = np.asarray(nodes,dtype=float)
  if vert is None:
    vert = np.zeros((0,nodes.shape[1]),dtype=float)
  else:  
    vert = np.asarray(vert,dtype=float)
  
  if smp is None:
    smp = np.zeros((0,nodes.shape[1]),dtype=int)
  else:
    smp = np.asarray(smp,dtype=int)
    
  if bound_force:
    bound_vert,bound_smp = vert,smp
  else:
    bound_vert,bound_smp = None,None

  if rho is None:
    def rho(p):
      return np.ones(p.shape[0])
  
  if fix_nodes is None:
    fix_nodes = np.zeros((0,nodes.shape[1]),dtype=float)
  else:
    fix_nodes = np.asarray(fix_nodes,dtype=float)
      
  if m is None:
    # number of neighbors defaults to 3 raised to the number of 
    # spatial dimensions
    m = 3**nodes.shape[1]

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  m = min(m,nodes.shape[0]+fix_nodes.shape[0])
  # node positions after repulsion 
  out = _disperse(nodes,fix_nodes,rho,m,delta,bound_vert,bound_smp)
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

  
def menodes(N,vert,smp,rho=None,fix_nodes=None,
            itr=100,m=None,delta=0.05,
            sort_nodes=True,bound_force=False):
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

  fix_nodes : (F,D) array, optional
    Nodes which do not move and only provide a repulsion force.
 
  itr : int, optional
    Number of repulsion iterations. If this number is small then the 
    nodes will not reach a minimum energy equilibrium.

  m : int, optional
    Number of neighboring nodes to use when calculating the repulsion 
    force. When *m* is small, the equilibrium state tends to be a 
    uniform node distribution (regardless of *rho*), when *m* is 
    large, nodes tend to get pushed up against the boundaries.

  delta : float, optional
    Scaling factor for the node step size in each iteration. The 
    step size is equal to *delta* times the distance to the nearest 
    neighbor.

  sort_nodes : bool, optional
    If True, nodes that are close in space will also be close in 
    memory. This is done with the Reverse Cuthill-McKee algorithm.
      
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
  It is assumed that *vert* and *smp* define a closed domain. If 
  this is not the case, then it is likely that an error message will 
  be raised which says "ValueError: No intersection found for 
  segment ...".
      
  '''
  logger.debug('starting minimum energy node generation') 
  max_sample_size = 1000000
  vert = np.asarray(vert,dtype=float) 
  smp = np.asarray(smp,dtype=int) 
  if rho is None:
    def rho(p):
      return np.ones(p.shape[0])

  if m is None:
    # number of neighbors defaults to 3 raised to the number of 
    # spatial dimensions
    m = 3**vert.shape[1]

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
      # estimated number of samples needed to get N accepted nodes
      sample_size = int(np.ceil((N-nodes.shape[0])/acceptance))
      # dont let sample_size exceed max_sample_size
      sample_size = min(sample_size,max_sample_size)

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
    new_nodes = new_nodes[rho(new_nodes) > seq1d]
    # reject test points that are outside of the domain
    new_nodes = new_nodes[contains(new_nodes,vert,smp)]
    # append to collection of accepted nodes
    nodes = np.vstack((nodes,new_nodes))
    logger.debug('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]
  # use a minimum energy algorithm to spread out the nodes
  for i in range(itr):
    logger.debug('starting node repulsion iteration %s of %s' % (i+1,itr)) 
    nodes = disperse(nodes,vert=vert,smp=smp,rho=rho,
                     fix_nodes=fix_nodes,m=m,
                     delta=delta,bound_force=bound_force)

  logger.debug('snapping nodes to boundary') 
  nodes,smpid = snap_to_boundary(nodes,vert,smp,delta=0.5)
  if sort_nodes:
    # sort so that nodes that are close in space are also close in 
    # memory
    idx = neighbor_argsort(nodes,m=m)
    nodes = nodes[idx]
    smpid = smpid[idx] 
  
  logger.debug('finished generating nodes') 
  return nodes,smpid
