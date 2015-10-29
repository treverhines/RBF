# distutils: extra_compile_args = -fopenmp  
# distutils: extra_link_args = -fopenmp 
from __future__ import division
import numpy as np
cimport numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.stdio cimport printf
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
  if (out.uv.x == 0.0) & (out.uv.y == 0):
    printf('WARNING: generated an edge with no length\n')

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

cdef bint veq(vector a, vector b) nogil:
  if (a.x == b.x) & (a.y == b.y):
    return True
  else:
    return False
  
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
cdef bint c_is_parallel(edge q,edge p) nogil:
  if cross(q.uv,p.uv) == 0.0:
    return True
  else:
    return False


# returns true if two edges intersect. An intersection is when any
# part of two segments meet at one points except for their tails
# i.e. they cannot intersect at t=0 or u=0. This means that edges
# connected from tip to tail will not be identified as intersecting    
cdef bint c_is_intersecting(edge q,edge p) nogil:
  cdef:
    vector qp = vsub(q.xy,p.xy)
    double a,b,c,t,u

  if c_is_parallel(q,p):
    # The only way they can intersect now is if they share endpoints
    if c_is_overlapping(q,p):
      return False     

    elif veq(vadd(q.xy,q.uv),vadd(p.xy,p.uv)):
      return True
   
    return False

  a = cross(qp,q.uv)
  b = cross(qp,p.uv)
  c = cross(p.uv,q.uv)
  t = a/c
  u = b/c
  if (t>0.0) & (t<=1.0) & (u>0.0) & (u<=1.0):
    return True

  else:
    return False

# returns true if two segments fall on the same line
cdef bint c_is_collinear(edge q,edge p) nogil:
  cdef:
    vector qp = vsub(q.xy,p.xy)

  if not (cross(qp,p.uv) == 0.0):
    return False

  if not c_is_parallel(q,p):
    return False

  else:
    return True
  
# returns true if there is some finite width where two vectors overlap
cdef bint c_is_overlapping(edge p,edge q) nogil: 
  cdef:
    vector qp = vsub(q.xy,p.xy)  
    vector pq = vsub(p.xy,q.xy) 
    double a,b,c,t0,t1

  if not c_is_collinear(p,q):
    return False

  a = dot(qp,p.uv)
  b = dot(q.uv,p.uv)
  c = dot(p.uv,p.uv)
  t0 = a/c
  t1 = t0 + b/c
  if ((t0 <= 0.0) & (t1 <= 0.0)) | ((t0 >= 1.0) & (t1 >= 1.0)):
    return False
  else:
    return True

# returns true if connections between the nodes produce a
# nonintersecting, nonoverlapping curve
def is_jordan(double[:,:] nodes):
  cdef:
    unsigned int r = nodes.shape[0]
    unsigned int i,j
    vector v1,v2,v3,v4
    edge e1,e2

  assert nodes.shape[1] == 2

  for i in range(r-1):
    for j in range(i):
      v1 = make_vector(nodes[i,0],nodes[i,1])
      v2 = make_vector(nodes[i+1,0],nodes[i+1,1])
      e1 = make_edge(v1,v2)              
      v3 = make_vector(nodes[j,0],nodes[j,1])
      v4 = make_vector(nodes[j+1,0],nodes[j+1,1])
      e2 = make_edge(v3,v4)
      if c_is_intersecting(e1,e2):
        return False               
      if c_is_overlapping(e1,e2):
        return False               

  for j in range(r-1):
    v1 = make_vector(nodes[r-1,0],nodes[r-1,1])
    v2 = make_vector(nodes[0,0],nodes[0,1])
    e1 = make_edge(v1,v2)              
    v3 = make_vector(nodes[j,0],nodes[j,1])
    v4 = make_vector(nodes[j+1,0],nodes[j+1,1])
    e2 = make_edge(v3,v4)
    if c_is_intersecting(e1,e2):
      return False               
    if c_is_overlapping(e1,e2):
      return False               

  return True

# returns true the point is within the nodes
@boundscheck(False)
@wraparound(False)
cdef bint cn_contains_k(double[:,:] vertices,
                        double[:] point, 
                        double[:] min_point) nogil:
  cdef:
    unsigned int r = vertices.shape[0]
    unsigned int i
    unsigned int count = 0
    edge e1,e2
    vector v1,v2,v3,v4 

  v1 = make_vector(point[0],point[1])
  v2 = make_vector(min_point[0],min_point[1])
  e1 = make_edge(v1,v2)
  for i in range(r-1):
    v3 = make_vector(vertices[i,0],vertices[i,1])
    v4 = make_vector(vertices[i+1,0],vertices[i+1,1])
    e2 = make_edge(v3,v4)
    if c_is_intersecting(e1,e2):
      count += 1

  v3 = make_vector(vertices[r-1,0],vertices[r-1,1])
  v4 = make_vector(vertices[0,0],vertices[0,1])  
  e2 = make_edge(v3,v4)

  if c_is_intersecting(e1,e2):
    count += 1

  return count%2  

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


def is_parallel(double[:,:] seg1,double[:,:] seg2):
  cdef:
    vector v1,v2,v3,v4
    edge e1,e2 

  assert seg1.shape[0] == 2
  assert seg2.shape[0] == 2
  assert seg1.shape[1] == 2
  assert seg2.shape[1] == 2

  v1 = make_vector(seg1[0,0],seg1[0,1]) 
  v2 = make_vector(seg1[1,0],seg1[1,1]) 
  e1 = make_edge(v1,v2)
  v3 = make_vector(seg2[0,0],seg2[0,1]) 
  v4 = make_vector(seg2[1,0],seg2[1,1]) 
  e2 = make_edge(v3,v4)

  return c_is_parallel(e1,e2)


def is_collinear(double[:,:] seg1,double[:,:] seg2):
  cdef:
    vector v1,v2,v3,v4
    edge e1,e2 

  assert seg1.shape[0] == 2
  assert seg2.shape[0] == 2
  assert seg1.shape[1] == 2
  assert seg2.shape[1] == 2

  v1 = make_vector(seg1[0,0],seg1[0,1]) 
  v2 = make_vector(seg1[1,0],seg1[1,1]) 
  e1 = make_edge(v1,v2)
  v3 = make_vector(seg2[0,0],seg2[0,1]) 
  v4 = make_vector(seg2[1,0],seg2[1,1]) 
  e2 = make_edge(v3,v4)

  return c_is_collinear(e1,e2)


def is_intersecting(double[:,:] seg1,double[:,:] seg2):
  cdef:
    vector v1,v2,v3,v4
    edge e1,e2 

  assert seg1.shape[0] == 2
  assert seg2.shape[0] == 2
  assert seg1.shape[1] == 2
  assert seg2.shape[1] == 2

  v1 = make_vector(seg1[0,0],seg1[0,1]) 
  v2 = make_vector(seg1[1,0],seg1[1,1]) 
  e1 = make_edge(v1,v2)
  v3 = make_vector(seg2[0,0],seg2[0,1]) 
  v4 = make_vector(seg2[1,0],seg2[1,1]) 
  e2 = make_edge(v3,v4)

  return c_is_intersecting(e1,e2)


def is_overlapping(double[:,:] seg1,double[:,:] seg2):
  cdef:
    vector v1,v2,v3,v4
    edge e1,e2 

  assert seg1.shape[0] == 2
  assert seg2.shape[0] == 2
  assert seg1.shape[1] == 2
  assert seg2.shape[1] == 2

  v1 = make_vector(seg1[0,0],seg1[0,1]) 
  v2 = make_vector(seg1[1,0],seg1[1,1]) 
  e1 = make_edge(v1,v2)
  v3 = make_vector(seg2[0,0],seg2[0,1]) 
  v4 = make_vector(seg2[1,0],seg2[1,1]) 
  e2 = make_edge(v3,v4)

  return c_is_overlapping(e1,e2)
  
