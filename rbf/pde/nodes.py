'''
provides a function for generating a locally quasi-uniform
distribution of nodes over an arbitrary domain
'''
from __future__ import division
import logging

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from rbf.utils import assert_shape
from rbf.pde.knn import k_nearest_neighbors
from rbf.pde.sampling import rejection_sampling, poisson_discs
from rbf.pde.geometry import (intersection,
                              intersection_count,
                              simplex_outward_normals,
                              simplex_normals,
                              nearest_point,
                              oriented_simplices)

logger = logging.getLogger(__name__)


def _disperse(nodes,
              rho=None,
              pinned_nodes=None,
              m=None,
              delta=0.1,
              vert=None,
              smp=None):
  '''
  Returns the new position of the free nodes after a dispersal step.
  Nodes on opposite sides of the boundary defined by `vert` and `smp`
  cannot repel eachother. This does not handle node intersections with
  the boundary
  '''
  if rho is None:
    rho = lambda x: np.ones(x.shape[0])

  if pinned_nodes is None:
    pinned_nodes = np.zeros((0, nodes.shape[1]), dtype=float)

  pinned_nodes = np.asarray(pinned_nodes, dtype=float)
  if m is None:
    # the default number of neighboring nodes to use when computing
    # the repulsion force is 7 for 2D and 13 for 3D
    if nodes.shape[1] == 2:
      m = 7

    elif nodes.shape[1] == 3:
      m = 13

  # ensure that the number of nodes used to determine repulsion force
  # is less than or equal to the total number of nodes
  m = min(m, nodes.shape[0]+pinned_nodes.shape[0])
  # if m is 0 or 1 then the nodes remain stationary
  if m <= 1:
    return np.array(nodes, copy=True)

  # form collection of all nodes
  all_nodes = np.vstack((nodes, pinned_nodes))
  # find index and distance to nearest nodes
  i, d = k_nearest_neighbors(nodes, all_nodes, m, vert=vert, smp=smp)
  # dont consider a node to be one of its own nearest neighbors
  i, d = i[:, 1:], d[:, 1:]
  # compute the force proportionality constant between each node
  # based on their charges
  c = 1.0/(rho(all_nodes)[i, None]*rho(nodes)[:, None, None])
  # calculate forces on each node resulting from the `m` nearest
  # nodes.
  forces = c*(nodes[:, None, :] - all_nodes[i, :])/d[:, :, None]**3
  # sum up all the forces for each node
  direction = np.sum(forces, axis=1)
  # normalize the net forces to one
  direction /= np.linalg.norm(direction, axis=1)[:, None]
  # in the case of a zero vector replace nans with zeros
  direction = np.nan_to_num(direction)
  # move in the direction of the force by an amount proportional to
  # the distance to the nearest neighbor
  step = delta*d[:, 0, None]*direction
  # new node positions
  out = nodes + step
  return out


def _disperse_within_boundary(nodes,
                              vert,
                              smp,
                              rho=None,
                              pinned_nodes=None,
                              m=None,
                              delta=0.1,
                              bound_force=False):
  '''
  Returns `nodes` after beingly slightly dispersed within the
  boundaries defined by `vert` and `smp`. The disperson is analogous
  to electrostatic repulsion, where neighboring node exert a repulsive
  force on eachother. If a node is repelled into a boundary then it
  bounces back in.
  '''
  if bound_force:
    bound_vert, bound_smp = vert, smp
  else:
    bound_vert, bound_smp = None, None

  # node positions after repulsion
  out = _disperse(nodes, rho=rho, pinned_nodes=pinned_nodes, m=m,
                  delta=delta, vert=bound_vert, smp=bound_smp)
  # boolean array of nodes which are now outside the domain
  crossed = intersection_count(nodes, out, vert, smp) > 0
  # point where nodes intersected the boundary and the simplex they
  # intersected at
  intr_pnt, intr_idx = intersection(nodes[crossed], out[crossed], vert, smp)
  # normal vector to intersection point
  intr_norms = simplex_normals(vert, smp[intr_idx])
  # distance that the node wanted to travel beyond the boundary
  res = out[crossed] - intr_pnt
  # bouce node off the boundary
  out[crossed] -= 2*intr_norms*np.sum(res*intr_norms, 1)[:, None]
  # check to see if the bounced nodes still intersect the boundary. If
  # not then set the bounced nodes back to their original position
  crossed = intersection_count(nodes, out, vert, smp) > 0
  out[crossed] = nodes[crossed]
  return out


def _snap_to_boundary(nodes, vert, smp, delta=0.5):
  '''
  Snaps nodes to the boundary defined by `vert` and `smp`. This is
  done by slightly shifting each node along the basis directions and
  checking to see if that caused a boundary intersection. If so, then
  the intersecting node is reset at the point of intersection.

  Returns
  -------
  (N, D) float array
    New nodes positions.

  (N, D) int array
    Index of the simplex that each node is on. If a node is not on a
    simplex (i.e. it is an interior node) then the simplex index is
    -1.

  '''
  n, dim = nodes.shape
  # find the distance to the nearest node
  dx = k_nearest_neighbors(nodes, nodes, 2)[1][:, 1]
  nrst_pnt, nrst_smpid = nearest_point(nodes, vert, smp)
  snap = np.linalg.norm(nrst_pnt - nodes, axis=1) < dx*delta
  out_smpid = np.full(n, -1, dtype=int)
  out_nodes = np.array(nodes, copy=True)
  out_nodes[snap] = nrst_pnt[snap]
  out_smpid[snap] = nrst_smpid[snap]
  return out_nodes, out_smpid


def _make_normal_vectors(smpid, vert, smp):
  '''
  Create an (n, d) array of normal vectors for each node. If the node
  is not a boundary node then the corresponding row contains NaN's.
  '''
  # get the normal vectors for each simplex
  simplex_normals = simplex_outward_normals(vert, smp)
  # allocate an array of nans for the node normal vectors 
  normals = np.full((smpid.shape[0], vert.shape[1]), np.nan)
  # find which nodes are attached to a simplex. set those normal
  # vectors to be the normal vector for the simplex they are attached
  # to
  normals[smpid >= 0] = simplex_normals[smpid[smpid >= 0]]
  return normals


def _make_group_indices(smpid, boundary_groups):
  '''
  Create a dictionary identifying the nodes in each group.
  '''
  groups = {'interior': [-1]}
  # each boundary group name is prepended with "boundary:" 
  for k, v in boundary_groups.items():
    groups['boundary:%s' % k] =  v
    
  # put the nodes into groups based on which (if any) simplex the
  # nodes are attached to.
  indices = {}
  for group_name, group_smp in groups.items():
    # get the node indices that belong to this group
    group_indices = [i for i, s in enumerate(smpid) if s in group_smp]
    indices[group_name] = np.array(group_indices, dtype=int)

  return indices


def _vertex_outward_normals(vert, smp):
  '''
  Get the "normal" vectors for each vertex of the domain. Here the
  normal vector is the average of the normal vectors for each simplex
  that the vertex is part of.
  '''
  simplex_normals = simplex_outward_normals(vert, smp)
  vertex_normals = np.zeros_like(vert)
  for i, s in enumerate(smp):
    vertex_normals[s] += simplex_normals[i]
  
  vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, None]   
  return vertex_normals


def _append_vertices(nodes,
                     groups, 
                     normals,
                     vert,
                     smp,
                     boundary_groups):
  '''
  Append the domain vertices to the node set. 
  '''
  # append the vertices
  out_nodes = np.vstack((nodes, vert))
  # append the vertex normals
  vertex_normals = _vertex_outward_normals(vert, smp)
  out_normals = np.vstack((normals, vertex_normals))
  # find the first simplex containing each vertex
  smpid = [np.nonzero(smp == i)[0][0] for i in range(vert.shape[0])] 
  # find out which group each vertex belongs to
  vertex_groups = _make_group_indices(smpid, boundary_groups)
  out_groups = groups.copy()
  for group_name, group_idx in vertex_groups.items():  
    # append the vertex groups to each group
    group_idx = group_idx + nodes.shape[0]
    group_idx = np.hstack((groups[group_name], group_idx))
    out_groups[group_name] = group_idx
              
  return out_nodes, out_groups, out_normals      


def _append_ghost_nodes(nodes, 
                        groups,
                        normals,
                        boundary_groups_with_ghosts):
  '''                        
  add ghost nodes for the specified groups. This is smart enough to
  not create duplicate ghost nodes if the boundary groups share
  simplices
  '''
  # collect the indices of all nodes that get a ghost
  idx = set()
  for k in boundary_groups_with_ghosts:
    idx = idx.union(groups['boundary:%s' % k])
  
  idx = list(idx)      
  # get the distance to the nearest neighbor for these nodes
  dx = k_nearest_neighbors(nodes[idx], nodes, 2)[1][:, 1]
  # create ghost nodes for this group
  ghosts = nodes[idx] + dx[:, None]*normals[idx]
  # append the ghosts to the nodes
  out_nodes = np.vstack((nodes, ghosts))
  # append a set of nans to the normals
  ghost_normals = np.full_like(ghosts, np.nan)
  out_normals = np.vstack((normals, ghost_normals))
  # find the indices of ghost nodes that belong to each boundary
  # group
  out_groups = groups.copy()
  for k in boundary_groups_with_ghosts:
    ghost_group_k = []
    for n in groups['boundary:%s' % k]:
      # find the index of the ghost node corresponding to node `n`
      g = nodes.shape[0] + idx.index(n)
      ghost_group_k.append(g)
    
    out_groups['ghosts:%s' % k] = np.array(ghost_group_k, dtype=int)    

  # add the indices for the remaining groups  
  return out_nodes, out_groups, out_normals


def _neighbor_argsort(nodes, m=None):
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
  >>> nodes = np.array([[0.0, 1.0],
                        [2.0, 1.0],
                        [1.0, 1.0]])
  >>> idx = neighbor_argsort(nodes, 2)
  >>> nodes[idx]
  array([[ 2.,  1.],
         [ 1.,  1.],
         [ 0.,  1.]])

  '''
  if m is None:
    # this should be roughly equal to the stencil size for the RBF-FD
    # problem
    m = 5**nodes.shape[1]

  m = min(m, nodes.shape[0])
  # find the indices of the nearest m nodes for each node
  idx = k_nearest_neighbors(nodes, nodes, m)[0]
  # efficiently form adjacency matrix
  col = idx.ravel()
  row = np.repeat(np.arange(nodes.shape[0]), m)
  data = np.ones(nodes.shape[0]*m, dtype=bool)
  mat = csc_matrix((data, (row, col)), dtype=bool)
  permutation = reverse_cuthill_mckee(mat)
  return permutation


def _sort_nodes(nodes, groups, normals):
  '''
  sort so that nodes that are close in space are also close in memory
  '''
  sort_idx = _neighbor_argsort(nodes)
  out_nodes = nodes[sort_idx]
  out_normals = normals[sort_idx]
  # update the groups because of this sorting
  out_groups = {}
  reverse_sort_idx = np.argsort(sort_idx)
  for key, val in groups.items():
    out_groups[key] = reverse_sort_idx[val]

  return out_nodes, out_groups, out_normals


def _test_node_spacing(nodes, rho):
  '''
  Test that no nodes are unusually close to eachother (which may
  have occurred when snapping nodes to the boundary or placing ghost
  nodes.
  '''
  if rho is None:
    rho = lambda x: np.ones(x.shape[0])

  # distance to nearest neighbor
  dist = k_nearest_neighbors(nodes, nodes, 2)[1][:, 1]
  if np.any(dist == 0.0):
    is_zero = (dist == 0.0)
    indices, = is_zero.nonzero()
    for idx in indices:
      logger.warning(
        'Node %s (%s) is in the same location as another node.' 
        % (idx, nodes[idx]))
    
  density = 1.0/dist**nodes.shape[1]
  normalized_density = np.log10(density / rho(nodes))
  percs = np.percentile(normalized_density, [10, 50, 90])
  med = percs[1]
  idr = percs[2] - percs[0]
  is_too_close = normalized_density < (med - 2*idr)
  if np.any(is_too_close):
    indices, = is_too_close.nonzero()
    for idx in indices:
      logger.warning(
        'Node %s (%s) is unusually close to a neighboring '
        'node.' % (idx, nodes[idx]))


def min_energy_nodes(N, vert, smp,
                     rho=None,
                     pinned_nodes=None,
                     itr=100,
                     m=None,
                     delta=0.05,
                     snap_delta=0.5,
                     boundary_groups=None,
                     boundary_groups_with_ghosts=None,
                     include_vertices=False,
                     bound_force=False):
  '''
  Generates nodes within a 1, 2, or 3 dimensional domain using a
  minimum energy algorithm.

  The algorithm is as follows: A quasi-random set of nodes is first
  generated within the domain from a Halton sequence. The nodes
  positions are then iteratively adjusted. For each iteration, the
  nearest neighbors to each node are found. A repulsion force is
  calculated for each node using the distance to its nearest neighbors
  and their charges (which are inversely proportional to the node
  density). Each node then moves in the direction of the net force
  acting on it. If a node is repelled into boundary, it will bounce
  back into the domain. When the iteration are complete, nodes that
  are sufficiently close to the boundary are snapped to the boundary.
  This function returns the nodes, the normal vectors to the boundary
  nodes, and an index set indicating which nodes belong to which
  group.

  Parameters
  ----------
  N : int
    Number of nodes

  vert : (P, D) array
    Vertices making up the boundary

  smp : (Q, D) array
    Describes how the vertices are connected to form the boundary

  rho : function, optional
    Node density function. Takes a (?, D) array of coordinates in D
    dimensional space and returns an (?,) array of densities which
    have been normalized so that the maximum density in the domain is
    1.0. This function will still work if the maximum value is
    normalized to something less than 1.0; however it will be less
    efficient.

  pinned_nodes : (F, D) array, optional
    Nodes which do not move and only provide a repulsion force. These
    nodes are included in the set of nodes returned by this function
    and they are in the group named "pinned".

  itr : int, optional
    Number of repulsion iterations. If this number is small then the
    nodes will not reach a minimum energy equilibrium.

  m : int, optional
    Number of neighboring nodes to use when calculating the repulsion
    force. This defaults to 7 for 2D nodes and 13 for 3D nodes.
    Deviating from these default values may yield a node distribution
    that is not consistent with the node density function `rho`. 

  delta : float, optional
    Scaling factor for the node step size in each iteration. The
    step size is equal to `delta` times the distance to the nearest
    neighbor.

  snap_delta : float, optional
    Controls the maximum snapping distance. The maximum snapping
    distance for each node is `snap_delta` times the distance to the
    nearest neighbor. This defaults to 0.5.

  boundary_groups: dict, optional 
    Dictionary defining the boundary groups. The keys are the names of
    the groups and the values are lists of simplex indices making up
    each group. This function will return a dictionary identifying
    which nodes belong to each boundary group. By default, there is a
    group for each simplex making up the boundary and another group
    named 'all' for the entire boundary. Specifically, The default
    value is `{'all':range(len(smp)), '0':[0], '1':[1], ...}`.

  boundary_groups_with_ghosts: list of strs, optional
    List of boundary groups that will be given ghost nodes. By
    default, no boundary groups are given ghost nodes. The groups
    specified here must exist in `boundary_groups`.

  bound_force : bool, optional
    If `True`, then nodes cannot repel other nodes through the domain
    boundary. Set this to `True` if the domain has edges that nearly
    touch eachother. Setting this to `True` may significantly increase
    computation time.

  include_vertices : bool, optional
    If `True`, then the vertices will be included in the output nodes.
    Each vertex will be assigned to the boundary group that its
    adjoining simplices are part of. If the simplices are in multiple
    groups, then the vertex will be assigned to the group containing
    the simplex that comes first in `smp`.

  Returns
  -------
  (N, D) float array
    Nodes positions

  dict 
    The indices of nodes belonging to each group. There will always be
    a group called 'interior' containing the nodes that are not on the
    boundary. By default there is a group containing all the boundary
    nodes called 'boundary:all', and there are groups containing the
    boundary nodes for each simplex called 'boundary:0', 'boundary:1',
    ..., 'boundary:Q'. If `boundary_groups` was specified, then those
    groups will be included in this dictionary and their names will be
    given a 'boundary:' prefix. If `boundary_groups_with_ghosts` was
    specified then those groups of ghost nodes will be included in
    this dictionary and their names will be given a 'ghosts:' prefix.
    
  (N, D) float array
    Outward normal vectors for each node. If a node is not on the
    boundary then its corresponding row will contain NaNs.

  Notes
  -----
  It is assumed that `vert` and `smp` define a closed domain. If
  this is not the case, then it is likely that an error message will
  be raised which says "ValueError: No intersection found for
  segment ...".

  Examples
  --------
  make 9 nodes within the unit square   

  >>> vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  
  >>> smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
  >>> out = min_energy_nodes(9, vert, smp)  

  view the nodes

  >>> out[0]  
  array([[ 0.50325675,  0.        ],
         [ 0.00605261,  1.        ],
         [ 1.        ,  0.51585247],
         [ 0.        ,  0.00956821],
         [ 1.        ,  0.99597894],
         [ 0.        ,  0.5026365 ],
         [ 1.        ,  0.00951112],
         [ 0.48867638,  1.        ],
         [ 0.54063894,  0.47960892]])

  view the indices of nodes making each group

  >>> out[1] 
  {'boundary:0': array([0]),
   'boundary:1': array([6, 4, 2]),
   'boundary:2': array([7, 1]),
   'boundary:3': array([5, 3]),
   'boundary:all': array([7, 6, 5, 4, 3, 2, 1, 0]),
   'interior': array([8])}

  view the outward normal vectors for each node, note that the normal
  vector for the interior node is `nan`

  >>> out[2] 
  array([[  0.,  -1.],
         [  0.,   1.],
         [  1.,  -0.],
         [ -1.,  -0.],
         [  1.,  -0.],
         [ -1.,  -0.],
         [  1.,  -0.],
         [  0.,   1.],
         [ nan,  nan]])
    
  '''
  logger.debug('starting minimum energy node generation')
  vert = np.asarray(vert, dtype=float)
  assert_shape(vert, (None, None), 'vert')
  
  smp = np.asarray(smp, dtype=int)
  assert_shape(smp, (None, vert.shape[1]), 'smp')
  
  # make sure the simplex normal vectors point outward    
  smp = oriented_simplices(vert, smp)
  
  if boundary_groups is None:
    boundary_groups = {'all': range(smp.shape[0])}
    for i in range(smp.shape[0]):
      boundary_groups[str(i)] = [i]
    
  if pinned_nodes is None:
    pinned_nodes = np.zeros((0, vert.shape[1]), dtype=float)
  else:
    pinned_nodes = np.array(pinned_nodes, dtype=float)            
    
  assert_shape(pinned_nodes, (None, vert.shape[1]), 'pinned_nodes')
  
  logger.debug('finding node positions with rejection sampling')
  nodes = rejection_sampling(N, rho, vert, smp)
    
  # `pinned_nodes` consist of specific nodes that we want included in
  # the output nodes. If `include_vertices` is True then add the
  # vertices to the pinned nodes, labeling the combination as
  # `pinned_nodes_`
  if include_vertices:  
    pinned_nodes_ = np.vstack((pinned_nodes, vert))
  else:
    pinned_nodes_ = pinned_nodes
      
  # use a minimum energy algorithm to spread out the nodes
  for i in range(itr):
    logger.debug('starting node repulsion iteration %s of %s' % (i+1, itr))
    nodes = _disperse_within_boundary(
      nodes, vert, smp, rho=rho, pinned_nodes=pinned_nodes_, m=m, 
      delta=delta, bound_force=bound_force)

  nodes, smpid = _snap_to_boundary(nodes, vert, smp, delta=snap_delta)
  normals = _make_normal_vectors(smpid, vert, smp)
  groups = _make_group_indices(smpid, boundary_groups)
  
  if include_vertices:
    nodes, groups, normals = _append_vertices(
      nodes, groups, normals, vert, smp, boundary_groups)

  if pinned_nodes.size != 0:
    # append the pinned nodes to the output    
    groups['pinned'] = np.arange(
      nodes.shape[0],
      nodes.shape[0] + pinned_nodes.shape[0])
    normals = np.vstack((
      normals, 
      np.full_like(pinned_nodes, np.nan)))        
    nodes = np.vstack((nodes, pinned_nodes))
            
  if boundary_groups_with_ghosts is not None:  
    nodes, groups, normals = _append_ghost_nodes(
      nodes, groups, normals, boundary_groups_with_ghosts)

  # sort `nodes` so that spatially adjacent nodes are close together
  # in memory. Update `indices` so that it is still pointing to the
  # same nodes
  nodes, groups, normals = _sort_nodes(nodes, groups, normals)

  # verify that the nodes are not too close to eachother
  _test_node_spacing(nodes, rho)

  logger.debug('finished generating %s nodes' % nodes.shape[0])
  return nodes, groups, normals




