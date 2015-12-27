#!/usr/bin/env python
from __future__ import division
import numpy as np
import rbf.nodegen
import rbf.normalize
from itertools import combinations


def edges(smp):
  '''                     
  returns the simplices for the edges of the manifold described by smp 
  '''
  smp = np.asarray(smp)
  dim = smp.shape[1]
  out = []
  sub_smp = []
  for s in smp:
    for c in combinations(s,dim-1):
      c_list = list(c)
      c_list.sort()
      sub_smp.append(c_list)

  for s in sub_smp:
    count = 0
    for t in sub_smp:
      if s == t:
        count += 1

    if count == 1:
      out.append(s)

  return np.array(out)


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

def nodes_on_simplex(rho,vert):
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

  # find the rotation matrix which rotates the simplex normal onto the
  # last axis
  R = simplex_rotation(vert)

  # _r denotes rotated values
  vert_r = np.einsum('ij,...j->...i',R,vert)
  const_r = vert_r[0,-1]

  # remove last dimension from the vertices 
  vert_r = vert_r[:,:-1]
  
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
    lb = vert_r[0,0]
    ub = vert_r[1,0]
    nodes_r = rbf.nodegen.generate_nodes_1d(rho_r,lb,ub)
    N = nodes_r.shape[0]
    a = np.ones((N,1))*const_r
    nodes_r = np.hstack((nodes_r,a))
    nodes = np.einsum('ij,...j->...i',R.T,nodes_r)
   
    # find the normal vector to the simplex
    v1 = vert[1,:] - vert[0,:]
    normal = rbf.nodegen.normal(np.array([v1]))
    if normal[-1] < 0.0:
      normal *= -1

    normals = np.repeat(normal[None,:],N,axis=0)

    # make the group ids for the nodes
    groups = np.zeros(N)
    groups[nodes_r[:,0] == vert_r[0,0]] = 1
    groups[nodes_r[:,0] == vert_r[1,0]] = 2

  if dim == 3:
    smp_r = np.array([[0,1],[1,2],[2,0]])
    grp_r = np.array([1,2,3])
    nodes_r,norms_r,groups = rbf.nodegen.generate_nodes(
                               rho_r,vert_r,smp_r,
                               groups=grp_r)
    N = nodes_r.shape[0]
    a = np.ones((N,1))*const_r
    nodes_r = np.hstack((nodes_r,a))
    nodes = np.einsum('ij,...j->...i',R.T,nodes_r)

    # find the normal vector to the simplex
    v1 = vert[1,:] - vert[0,:]
    v2 = vert[2,:] - vert[0,:]
    normal = rbf.nodegen.normal(np.array([v1,v2]))
    if normal[-1] < 0.0:
      normal *= -1

    normals = np.repeat(normal[None,:],N,axis=0)


  return nodes,normals,groups


