# distutils: extra_compile_args = -fopenmp  
# distutils: extra_link_args = -fopenmp 
from __future__ import division
import numpy as np
cimport numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.stdio cimport printf
from cython cimport boundscheck,wraparound


cdef double _det(double a,
                  double b,
                  double c,
                  double d) nogil:
  return a*d - b*c


cdef bint _is_collinear(double x00,double y00,
                        double x01,double y01,
                        double x10,double y10,
                        double x11,double y11) nogil:
  cdef:
    double a,b
  
  a =  _det(x10-x00,x01,y10-y00,y01) 
  b =  _det(x11,x01,y11,y01) 

  if not (a == 0.0):
    return False

  if not (b == 0.0):
    return False

  else:
    return True


cdef bint _is_overlapping(double x00,double y00,
                           double x01,double y01,
                           double x10,double y10,
                           double x11,double y11) nogil:
  cdef:
    double a,b,c,t0,t1

  if not _is_collinear(x00,y00,
                       x01,y01,
                       x10,y10,
                       x11,y11):
    return False

  #a = dot(qp,p.uv)
  a = (x10 - x00)*x01 + (y10 - y00)*y01 
  #b = dot(q.uv,p.uv)
  b = x11*x01 + y11*y01
  #c = dot(p.uv,p.uv)
  c = x01*x01 + y01*y01
  t0 = a/c
  t1 = t0 + b/c
  if ((t0 <= 0.0) & (t1 <= 0.0)) | ((t0 >= 1.0) & (t1 >= 1.0)):
    return False

  else:
    return True

# returns true if two edges intersect. An intersection is when any
# part of two segments meet at one points except for their tails
# i.e. they cannot intersect at t=0 or u=0. This means that edges
# connected from tip to tail will not be identified as intersecting    
cdef bint _is_intersecting(double x00,double y00,
                            double x01,double y01,
                            double x10,double y10,
                            double x11,double y11,
                            bint anchor=True,
                            bint tip=True) nogil:
  cdef:
    double s,t,d
  
  d = _det(x11,x01,y11,y01)
  if d == 0.0:
    if _is_overlapping(x00,y00,
                       x01,y01,
                       x10,y10,
                       x11,y11):
      return False

    elif (x00 == x10) & (y00 == y10):
      s = 0.0
      t = 0.0

    elif (x00 == (x10+x11)) & (y00 == (y10+y11)):
      s = 1.0
      t = 0.0

    elif ((x00+x01) == (x10+x11)) & ((y00+y01) == (y10+y11)):
      s = 1.0
      t = 1.0

    elif ((x00+x01) == x10) & ((y00+y01) == y10):
      s = 0.0
      t = 1.0

    else:
      return False 

  else:
    s = (1/d)*((x00-x10)*y01 - (y00-y10)*x01)
    t = (1/d)*((x00-x10)*y11 - (y00-y10)*x11)  

  if anchor & tip: 
    if (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0):
      return True   

  if (not anchor) & tip: 
    if (s > 0.0) & (s <= 1.0) & (t > 0.0) & (t <= 1.0):
      return True   

  if anchor & (not tip): 
    if (s >= 0.0) & (s < 1.0) & (t >= 0.0) & (t < 1.0):
      return True   

  if (not anchor) & (not tip): 
    if (s > 0.0) & (s < 1.0) & (t > 0.0) & (t < 1.0):
      return True   

  return False


# returns true the point is within the nodes
@boundscheck(False)
@wraparound(False)
cdef bint _cn_contains_k(double[:] point, 
                         double[:,:] vertices,
                         double[:] min_point) nogil:
  cdef:
    unsigned int r = vertices.shape[0]
    unsigned int i
    unsigned int count = 0

  for i in range(r-1):
    if _is_intersecting(vertices[i,0],vertices[i,1], 
                         vertices[i+1,0] - vertices[i,0],
                         vertices[i+1,1] - vertices[i,1],
                         point[0],point[1], 
                         min_point[0] - point[0],
                         min_point[1] - point[1],
                         anchor=False,tip=True):
      count += 1

  if _is_intersecting(vertices[r-1,0],vertices[r-1,1], 
                       vertices[0,0] - vertices[r-1,0],
                       vertices[0,1] - vertices[r-1,1],
                       point[0],point[1], 
                       min_point[0] - point[0],
                       min_point[1] - point[1],
                       anchor=False,tip=True):
    count += 1

  return count%2  


cdef double _is_left(double[:] P0,
                     double[:] P1,
                     double[:] P2) nogil:
  return ((P1[0] - P0[0])*(P2[1] - P0[1]) -
          (P2[0] - P0[0])*(P1[1] - P0[1]))


@boundscheck(False)
@wraparound(False)
cdef bint _wn_contains_k(double[:] point,
                         double[:,:] vertices) nogil:
  cdef:
    int wn = 0
    int i
    int r = vertices.shape[0]

  for i in range(r-1):
    if vertices[i,1] <= point[1]:
      if vertices[i+1,1]  > point[1]:
        if _is_left(vertices[i],vertices[i+1],point) > 0:
          wn += 1
    else:
      if (vertices[i+1,1]  <= point[1]):
        if _is_left(vertices[i],vertices[i+1],point) < 0:
          wn -= 1

  if vertices[r-1,1] <= point[1]:
    if vertices[0,1]  > point[1]:
      if _is_left(vertices[r-1],vertices[0],point) > 0:
        wn += 1
  else:
    if (vertices[0,1]  <= point[1]):
      if _is_left(vertices[r-1],vertices[0],point) < 0:
        wn -= 1

  return wn != 0


def intersects(double[:,:] vertices1,
               double[:,:] vertices2):
  cdef:
    unsigned int i,j
    unsigned int m = vertices1.shape[0]
    unsigned int n = vertices2.shape[0]

  assert vertices1.shape[1] == 2
  assert vertices2.shape[1] == 2

  for i in range(m-1):
    for j in range(n-1):
      if _is_intersecting(vertices1[i,0],vertices1[i,1], 
                          vertices1[i+1,0] - vertices1[i,0],
                          vertices1[i+1,1] - vertices1[i,1],
                          vertices2[j,0],vertices2[j,1], 
                          vertices2[j+1,0] - vertices2[j,0],
                          vertices2[j+1,1] - vertices2[j,1],
                          anchor=True,tip=True):
        return True

  return False


def contains(double[:,:] points,
             double[:,:] vertices):
  cdef:
    short[:] out = np.zeros(points.shape[0],dtype=np.int16) 
    double[:] min_point = np.min(vertices,0) - 1.0 
    unsigned int n = points.shape[0]
    unsigned int i 
    
  assert points.shape[1] == 2
  assert vertices.shape[1] == 2

  for i in prange(n,nogil=True):
    out[i] = _cn_contains_k(points[i,:],vertices,min_point)    
    #out[i] = _wn_contains_k(points[i,:],vertices)    

  return np.asarray(out,dtype=bool)

