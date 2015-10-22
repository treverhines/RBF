# distutils: extra_compile_args = -fopenmp                                                                         
# distutils: extra_link_args = -fopenmp 

from __future__ import division
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from cython cimport boundscheck,wraparound

# a vector consists of an x and y coordinate
cdef struct vector:
  double x
  double y

# an edge is a line segment define in terms of 
# a vector anchor (xy) and the distance from that anchor to the next
# point (uv)
cdef struct edge:
  vector xy
  vector uv

# create an edge in terms of two vectors 
cdef vector make_vector(double a,double b) nogil:
  cdef:
    vector out

  out.x = a
  out.y = b
  return out

# create an edge in terms of two vectors 
cdef edge make_edge(vector a,vector b) nogil:
  cdef:
    edge out

  out.xy = a
  out.uv = vsub(b,a)
  return out

# vector addition
cdef vector vadd(vector a, vector b) nogil:
  cdef:
    vector out

  out.x = a.x + b.x
  out.y = a.y + b.y
  return out
  
# vector subtraction
cdef vector vsub(vector a, vector b) nogil:
  cdef:
    vector out

  out.x = a.x - b.x
  out.y = a.y - b.y
  return out
  
# dot product
cdef double dot(vector a, vector b) nogil:
  cdef:
    double out

  out = a.x*b.x + a.y*b.y
  return out

# cross product
cdef double cross(vector a, vector b) nogil:
  cdef:
    double out

  out = a.x*b.y - a.y*b.x
  return out  

# returns true if two edges are parallel
cdef bint is_parallel(edge q,edge p) nogil:
  if cross(q.uv,p.uv) == 0.0:
    return True
  else:
    return False


# returns true if two edges intersect. An intersection is when any part 
# of two segments meet at one points. This does not work if segments
# are parallel and touching at an vertex
cdef bint is_intersecting(edge q,edge p) nogil:
  cdef:
    vector qp = vsub(q.xy,p.xy)
    vector pq = vsub(p.xy,q.xy)
    double a,b,c,t,u

  if is_parallel(q,p):
    return False

  a = cross(qp,q.uv)
  b = cross(qp,p.uv)
  c = cross(p.uv,q.uv)
  t = a/c
  u = -b/c
  if (t>=0.0) & (t<=1.0) & (u>=0.0) & (u<=1.0):
    return True

  else:
    return False

# returns true if two segments fall on the same line
cdef bint is_collinear(edge q,edge p) nogil:
  cdef:
    vector qp = vsub(q.xy,p.xy)

  if not (cross(qp,p.uv) == 0.0):
    return False

  if not is_parallel(q,p):
    return False

  else:
    return True
  
# returns true if there is some finite width where two vectors overlap
cdef bint is_overlapping(edge p,edge q):
  cdef:
    vector qp = vsub(q.xy,p.xy)  
    vector pq = vsub(p.xy,q.xy) 
    double a,b,c,d

  if not is_collinear(p,q):
    return False

  a = dot(qp,p.uv)
  b = dot(pq,q.uv)
  c = dot(p.uv,p.uv)
  d = dot(q.uv,q.uv)
  if (a > 0.0) & (a < c):
    return True  

  elif (b > 0.0) & (b < d):
    return True

  else:
    return False

# returns true the point is within the nodes
@boundscheck(False)
@wraparound(False)
cdef bint contains_k(double[:,:] nodes,
                     double[:] point, 
                     double[:] min_point) nogil:
  cdef:
    unsigned int r = nodes.shape[0]
    unsigned int i
    unsigned int count = 0
    edge e1,e2
    vector v1,v2,v3,v4 

  v1 = make_vector(point[0],point[1])
  v2 = make_vector(min_point[0],min_point[1])
  e1 = make_edge(v1,v2)
  for i in range(r-1):
    v3 = make_vector(nodes[i,0],nodes[i,1])
    v4 = make_vector(nodes[i+1,0],nodes[i+1,1])
    e2 = make_edge(v3,v4)
    if is_intersecting(e1,e2):
      count += 1

  v3 = make_vector(nodes[r-1,0],nodes[r-1,1])
  v4 = make_vector(nodes[0,0],nodes[0,1])  
  e2 = make_edge(v3,v4)
  if is_intersecting(e1,e2):
    count += 1

  return count%2  


#cpdef contains(np.ndarray[double,ndim=2] nodes,
#               np.ndarray[double,ndim=2] points):
@boundscheck(False)
@wraparound(False)
cpdef contains(double[:,:] nodes,
               double[:,:] points):
  cdef:
    short[:] out = np.zeros(points.shape[0],dtype=np.int16) 
    double[:] min_point = np.array([0.0,np.min(nodes[:,1])-1.0])
    unsigned int n = points.shape[0]
    unsigned int i 
    

  with nogil:
    for i in prange(n):
      out[i] = contains_k(nodes,points[i,:],min_point)    

  return np.asarray(out,dtype=bool)

def makeit():
  cdef:
    vector a = vector(x=1.,y=1.)
    vector b = vector(x=2.,y=2.)
    vector c = vector(x=0.,y=0.1)
    vector d = vector(x=1.0,y=1.0)
    edge A = make_edge(a=a,b=b)
    edge B = make_edge(a=d,b=c)

  print(is_collinear(A,B))
  print(is_parallel(A,B))
  print(is_intersecting(A,B))
  print(is_overlapping(B,A))
  #print(is_parallel(A,B))
  #print(is_intersecting(B,A))
  #print(is_collinear(B,A))
  #print(is_overlapping(B,A))
  
