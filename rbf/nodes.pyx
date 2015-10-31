# distutils: extra_compile_args = -fopenmp  
# distutils: extra_link_args = -fopenmp 
from __future__ import division
import numpy as np
import scipy.spatial
import rbf.halton
cimport numpy as np
cimport numpy as np
from cython.parallel cimport prange
from cython cimport boundscheck,wraparound


cdef double is_left(double[:] P0,
                    double[:] P1,
                    double[:] P2) nogil:
  return ((P1[0] - P0[0])*(P2[1] - P0[1]) -
          (P2[0] - P0[0])*(P1[1] - P0[1]))


@boundscheck(False)
@wraparound(False)
cdef bint wn_contains_k(double[:] point,
                        double[:,:] vertices) nogil:
  cdef:
    int wn = 0
    int i
    int r = vertices.shape[0]

  for i in range(r-1):
    if vertices[i,1] <= point[1]:
      if vertices[i+1,1]  > point[1]:
        if is_left(vertices[i],vertices[i+1],point) > 0:
          wn += 1
    else:
      if (vertices[i+1,1]  <= point[1]):
        if is_left(vertices[i],vertices[i+1],point) < 0:
          wn -= 1

  if vertices[r-1,1] <= point[1]:
    if vertices[0,1]  > point[1]:
      if is_left(vertices[r-1],vertices[0],point) > 0:
        wn += 1
  else:
    if (vertices[0,1]  <= point[1]):
      if is_left(vertices[r-1],vertices[0],point) < 0:
        wn -= 1

  return wn != 0

def contains(double[:,:] points,
             double[:,:] vertices):
  cdef:
    short[:] out = np.zeros(points.shape[0],dtype=np.int16) 
    unsigned int n = points.shape[0]
    unsigned int i 
    
  assert points.shape[1] == 2
  assert vertices.shape[1] == 2

  #if not is_jordan(vertices):
  #  raise ValueError, 'vertices produce overlapping or intersecting segments'    

  for i in prange(n,nogil=True):
    out[i] = wn_contains_k(points[i,:],vertices)    

  return np.asarray(out,dtype=bool)


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

def _repel(int_nodes,fix_nodes=None,itr=10,n=10,eps=0.2):
  tol = 1e-10
  int_nodes = np.copy(int_nodes)
  if fix_nodes is None:
    fix_nodes = np.zeros((0,int_nodes.shape[1]))

  fix_nodes = np.asarray(fix_nodes)
  nodes = np.vstack((int_nodes,fix_nodes))
  if n > nodes.shape[0]:
    n = nodes.shape[0]

  for k in range(itr):
    T = scipy.spatial.cKDTree(nodes)
    d,i = T.query(int_nodes,n)
    i = i[:,1:]
    d = d[:,1:]
    force = np.sum((int_nodes[:,None,:] - nodes[i,:])/d[:,:,None]**3,1)
    mag = np.linalg.norm(force,axis=1)
    idx = mag > tol
    force[idx] /= mag[idx,None]
    force[~idx] *= 0.0
    step = eps*d[:,0,None]*force
    int_nodes += step
    nodes = np.vstack((int_nodes,fix_nodes))

  return int_nodes        

def pick_nodes(N,lb,ub,bnd_nodes=None,bnd=None,rho=None,
               sample_size=None,
               repel_itr=10,repel_n=10,repel_eps=0.2):
  lb = np.asarray(lb)
  ub = np.asarray(ub)

  assert len(lb) == len(ub)
  assert np.all(ub > lb)
  if sample_size is None: 
    sample_size = N//2

  ndim = len(lb)
  H = rbf.halton.Halton(ndim+1)
  nodes = np.zeros((0,ndim))
  while nodes.shape[0] < N:
    size_start = nodes.shape[0]
    seqNd = H(sample_size)
    new_nodes = seqNd[:,:ndim]
    seq1d = seqNd[:,-1]
    new_nodes *= (ub - lb)
    new_nodes += lb
    new_nodes = _density_filter(new_nodes,seq1d,rho)
    nodes = np.vstack((nodes,new_nodes))
    nodes = _repel(nodes,bnd_nodes,
                   itr=repel_itr,
                   n=repel_n,
                   eps=repel_eps)
    nodes = _boundary_filter(nodes,bnd)    
    size_end = nodes.shape[0]
    acceptance = (size_end - size_start)/sample_size
    if acceptance <= 0:
      print('Warning: no samples are being retained')

  return nodes[:N]



    

  
