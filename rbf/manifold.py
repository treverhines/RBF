#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.nodegen
import rbf.normalize
from itertools import combinations


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
    normal = rbf.nodegen.normal(np.array([v1]))
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
    normal = rbf.nodegen.normal(np.array([v1,v2]))

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

def nodes_on_simplex(rho,vert,fix_nodes=None):
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
    normal = rbf.nodegen.normal(np.array([v1]))

  if dim == 3:
    smp_r = np.array([[0,1],[0,2],[1,2]])
    grp_r = np.array([1,2,3])
    # find the normal vector to the simplex
    v1 = vert[1,:] - vert[0,:]
    v2 = vert[2,:] - vert[0,:]
    normal = rbf.nodegen.normal(np.array([v1,v2]))

  nodes_r,norms_r,groups = rbf.nodegen.generate_nodes(
                             rho_r,vert_r,smp_r,
                             groups=grp_r,fix_nodes=fix_r)
  N = nodes_r.shape[0]
  a = np.ones((N,1))*const_r
  nodes_r = np.hstack((nodes_r,a))
  nodes = np.einsum('ij,...j->...i',R.T,nodes_r)

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


def nodes_on_surface(rho,vert,smp):
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
    # and it is necessary to make all vertices into fixed nodes
    fix_nodes = np.vstack((fix_nodes,v))

    n,m,g1 = nodes_on_simplex(rho,v,fix_nodes=fix_nodes)
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

  return nodes,normals,groups
