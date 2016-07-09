''' 
provides a function for generating a locally quasi-uniform 
distribution of nodes over an arbitrary domain
'''
from __future__ import division
import numpy as np
import rbf.halton
import rbf.geometry as gm
import rbf.integrate
import rbf.stencil
import logging
import scipy.sparse
logger = logging.getLogger(__name__)

def nearest_neighbor_argsort(nodes,n=10):
  ''' 
  Returns a pertumation array that sorts nodes so that neighboring 
  nodes are close together. This is done through use of a KD Tree and 
  the Reverse Cuthill-McKee algorithm

  Parameters
  ----------
    nodes : (N,D) array

    n : int 
      number of adjacencies to identify for each node. The permutation 
      array will place adjacent nodes close to eachother in memory.  
      This should be about equal to the stencil size for RBF-FD 
      method.

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


def check_node_spacing(rho,nodes,tol=0.25):
  ''' 
  Returns indices of nodes which are consistent with the node density 
  function rho. Indexing nodes with the output will return a pared 
  down node set that is consistent with rho

  Parameters
  ----------
    rho : function
      callable node density function
  
    nodes : (N,D) array

    tol : float
      if a nodes nearest neighbor deviates by more than this factor 
      from the expected node density, then the node is identified as 
      being too close

  Returns 
  ------- 
    keep_indices : (N,) int array
      array of node indices which are consistent with the node density 
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


def _repel_step(free_nodes,rho,
                fix_nodes,
                n,delta):
  ''' 
  returns the new position of the free nodes after a repulsion step
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)

  # if n is 0 or 1 then the nodes remain stationary
  if n <= 1:
    return free_nodes

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
  # caluclate forces on each node resulting from the *n* nearest nodes 
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
  free_nodes += step
  return free_nodes


def _repel_bounce(free_nodes,vert,
                  smp,rho,   
                  fix_nodes,
                  itr,n,delta,
                  max_bounces):
  ''' 
  nodes are repelled by eachother and bounce off boundaries
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)

  # this is used for the lengthscale of the domain
  scale = vert.ptp()

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  n = min(n,free_nodes.shape[0]+fix_nodes.shape[0])

  for k in range(itr):
    # node positions after repulsion 
    free_nodes_new = _repel_step(free_nodes,rho,
                                 fix_nodes,
                                 n,delta)

    # boolean array of nodes which are now outside the domain
    crossed = ~gm.contains(free_nodes_new,vert,smp)
    bounces = 0
    while np.any(crossed):
      # point where nodes intersected the boundary
      inter = gm.intersection_point(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vert,smp)

      # normal vector to intersection point
      norms = gm.intersection_normal(
                free_nodes[crossed],     
                free_nodes_new[crossed],
                vert,smp)
      
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
        crossed = ~gm.contains(free_nodes_new,vert,smp)
        bounces += 1

    free_nodes = free_nodes_new  
  
  return free_nodes


def _repel_stick(free_nodes,vert,
                 smp,rho,   
                 fix_nodes,
                 itr,n,delta):
  ''' 
  nodes are repelled by eachother and then become fixed when they hit 
  a boundary
  '''
  free_nodes = np.array(free_nodes,dtype=float,copy=True)

  # Keeps track of whether nodes in the interior or boundary. -1 
  # indicates interior and >= 0 indicates boundary. If its on the 
  # boundary then the number is the index of the simplex that the node 
  # is on
  smpid = np.repeat(-1,free_nodes.shape[0])

  # length scale of the domain
  scale = vert.ptp()

  # ensure that the number of nodes used to compute repulsion force is
  # less than or equal to the total number of nodes
  n = min(n,free_nodes.shape[0]+fix_nodes.shape[0])

  for k in range(itr):
    # indices of all interior nodes
    interior, = (smpid==-1).nonzero()

    # indices of nodes associated with a simplex (i.e. nodes which 
    # intersected a boundary
    boundary, = (smpid>=0).nonzero()

    # nodes which are stationary 
    all_fix_nodes = np.vstack((fix_nodes,free_nodes[boundary]))

    # new position of free nodes
    free_nodes_new = np.array(free_nodes,copy=True)
    # shift positions of interior nodes
    free_nodes_new[interior] = _repel_step(free_nodes[interior],
                                 rho,all_fix_nodes,n,delta)

    # indices of free nodes which crossed a boundary
    crossed = ~gm.contains(free_nodes_new,vert,smp)
  
    # if a node intersected a boundary then associate it with a simplex
    smpid[crossed] = gm.intersection_index(
                       free_nodes[crossed],     
                       free_nodes_new[crossed], 
                       vert,smp)

    # outward normal vector at intesection points
    norms = gm.intersection_normal(
              free_nodes[crossed],     
              free_nodes_new[crossed], 
              vert,smp)

    # intersection point for nodes which crossed a boundary
    inter = gm.intersection_point(
              free_nodes[crossed],     
              free_nodes_new[crossed],
              vert,smp)

    # new position of nodes which crossed the boundary is just within
    # the intersection point
    free_nodes_new[crossed] = inter - 1e-10*scale*norms
    free_nodes = free_nodes_new

  return free_nodes,smpid


def make_nodes(N,vert,smp,rho=None,fix_nodes=None,
               itr=20,neighbors=10,delta=0.1,
               sort_nodes=True):
  ''' 
  Generates nodes within the D-dimensional volume enclosed by the 
  simplexes using a minimum energy algorithm.  
  
  At each iteration the nearest neighbors to each node are found and 
  then a repulsion force is calculated using the distance to the 
  nearest neighbors and their charges (which is inversely proportional 
  to the node density). Each node then moves in the direction of the 
  net force acting on it.  The step size is equal to delta times the 
  distance to the nearest node. This is repeated for 2*n iterations.
  
  During the first *itr* iterations, if a node intersects a boundary 
  then it elastically bounces off the boundary. During the last *itr* 
  iterations, if a node intersects a boundary then it sticks to the 
  boundary at the intersection point.

  Paramters
  ---------
    N : int
      numbr of nodes
      
    vert : (P,D) array
      boundary vertices

    smp : (Q,D) array
      describes how the vertices are connected to form the boundary
    
    rho : function, optional
      node density function. Takes a (N,D) array of coordinates in D 
      dimensional space and returns an (N,) array of densities which 
      have been normalized so that the maximum density in the domain 
      is 1.0. This function will still work if the maximum value is 
      normalized to something less than 1.0; however it will be less 
      efficient.

    fix_nodes : (F,D) array, optional
      nodes which do not move and only provide a repulsion force
 
    itr : int, optional
      number of repulsion iterations. If this number is small then the 
      nodes will not reach a minimum energy equilibrium.

    neighbors : int, optional
      Number of neighboring nodes to use when calculating the 
      repulsion force. When *neighbors* is small, the equilibrium 
      state tends to be a uniform node distribution (regardless of 
      *rho*), when *neighbors* is large, nodes tend to get pushed up 
      against the boundaries.

    delta : float, optional
      Scaling factor for the node step size in each iteration. The 
      step size is equal to *delta* times the distance to the nearest 
      neighbor.

    sort_nodes : bool, optional
      If True, nodes that are close in space will also be close in 
      memory. This is done with the Reverse Cuthill-McKee algorithm

  Returns
  -------
    nodes: (N,D) float array 

    smpid: (N,) int array
      Index of the simplex that each node is on. If a node is not on a 
      simplex (i.e. it is an interior node) then the simplex index is 
      -1.

  Note
  ----
    It is assumed that *vert* and *smp* define a closed 
    domain. If this is not the case, the function will run normally 
    but the nodes will likely be greatly dispersed
    
  '''
  max_sample_size = 1000000

  vert = np.asarray(vert,dtype=float) 
  smp = np.asarray(smp,dtype=int) 
  if fix_nodes is None:
    fix_nodes = np.zeros((0,vert.shape[1]))
  else:
    fix_nodes = np.asarray(fix_nodes)

  if rho is None:
    def rho(p):
      return np.ones(p.shape[0])

  # form bounding box for the domain so that a RNG can produce values
  # that mostly lie within the domain
  lb = np.min(vert,axis=0)
  ub = np.max(vert,axis=0)
  ndim = vert.ndim
  
  # form Halton sequence generator
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
    new_nodes = new_nodes[gm.contains(new_nodes,vert,smp)]

    # append to collection of accepted nodes
    nodes = np.vstack((nodes,new_nodes))

    logger.info('accepted %s of %s nodes' % (nodes.shape[0],N))
    acceptance = nodes.shape[0]/cnt

  nodes = nodes[:N]

  # use a minimum energy algorithm to spread out the nodes
  logger.info('repelling nodes with boundary bouncing') 
  nodes = _repel_bounce(nodes,vert,smp,rho,
                        fix_nodes,itr,neighbors,delta,3)

  logger.info('repelling nodes with boundary sticking') 
  nodes,smpid = _repel_stick(nodes,vert,smp,rho,
                             fix_nodes,itr,neighbors,delta)

  # sort so that nodes that are close in space are also close in memory
  if sort_nodes:
    idx = nearest_neighbor_argsort(nodes,n=neighbors)
    nodes = nodes[idx]
    smpid = smpid[idx] 
  
  return nodes,smpid
