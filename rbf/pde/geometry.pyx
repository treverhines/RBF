'''
Module of cythonized functions for basic computational geometry in 2 and 3
dimensions. This modules requires geometric objects (e.g. polygons, polyhedra,
surfaces, segments, etc.) to be described as a collection of simplices (i.e.
segments or triangles). In this module, geometric objects in D-dimensional
space are described with an (N, D) array of vertices and and (M, D) array
describing the indices of vertices making up each simplex. As an example, the
unit square in two dimensions can be described as collection of line segments:

>>> vertices = [[0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]]
>>> simplices = [[0, 1],
                 [1, 2],
                 [2, 3],
                 [3, 0]]

A three dimensional cube can similarly be described as a collection of
triangles:

>>> vertices = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0]]
>>> simplices = [[1, 0, 4],
                 [5, 1, 4],
                 [7, 1, 5],
                 [3, 1, 7],
                 [0, 1, 3],
                 [2, 0, 3],
                 [0, 2, 6],
                 [4, 0, 6],
                 [5, 4, 7],
                 [4, 6, 7],
                 [2, 3, 7],
                 [6, 2, 7]]

This module is primarily used to find whether and where line segments intersect
a geometric object and whether points are contained within a closed geometric
object.  For example, one can determine whether a collection of points, saved
as `points`, are contained within a geometric object, defined by `vertices` and
`simplices` with the command

>>> contains(points, vertices, simplices)

which returns a boolean array.

One can find the number of times a collection of line segments, defined by
`start_points` and `end_points`, intersect a geometric object with the command

>> intersection_count(start_points, end_points, vertices, simplices)

which returns an array of the number of simplices intersected by each segment.
If it is known that a collection of line segments intersect a geometric object
then the intersection point can be found with the command

>> intersection(start_points, end_points, vertices, simplices)

This returns an (N, D) float array of intersection points, and an (N,) int
array identifying which simplex the intersection occurred at. If a line segment
does not intersect the geometric object then the above command returns a
ValueError. If there are multiple intersections for a single segment then only
the first detected intersection will be returned.

There are numerous other packages which can perform the same tasks as this
module. For example, Shapely can perform these 2-D operations (and more
robustly), but the point in polygon testing is too slow for RBF purposes. GDAL
is another option and that seems to now have python bindings that are
sufficiently fast. This module may eventually turn into a wrapper from some
GDAL functions.
'''
# python imports
from __future__ import division

import numpy as np
from scipy.special import factorial

from rbf.utils import assert_shape

# cython imports
from cython cimport boundscheck, wraparound, cdivision
from libc.math cimport fabs, sqrt, INFINITY


cdef struct vector1d:
    double x


cdef struct vector2d:
    double x
    double y


cdef struct vector3d:
    double x
    double y
    double z


cdef struct segment1d:
    vector1d a
    vector1d b


cdef struct segment2d:
    vector2d a
    vector2d b


cdef struct segment3d:
    vector3d a
    vector3d b


cdef struct triangle2d:
    vector2d a
    vector2d b
    vector2d c


cdef struct triangle3d:
    vector3d a
    vector3d b
    vector3d c


cdef double distance_1d(vector1d vec1, vector1d vec2) nogil:
    return fabs(vec1.x - vec2.x)


cdef double distance_2d(vector2d vec1, vector2d vec2) nogil:
    return sqrt((vec1.x - vec2.x)**2 + (vec1.y - vec2.y)**2)


cdef double distance_3d(vector3d vec1, vector3d vec2) nogil:
    return sqrt((vec1.x - vec2.x)**2 +
                (vec1.y - vec2.y)**2 +
                (vec1.z - vec2.z)**2)


@cdivision(True)
cdef vector2d orthogonal_2d(vector2d vec) nogil:
    '''
    Returns a normalized vector that is orthogonal to `vec`
    '''
    cdef:
        vector2d out
        float mag

    out.x = -vec.y
    out.y = vec.x
    mag = sqrt(out.x**2 + out.y**2)
    out.x /= mag
    out.y /= mag
    return out


@cdivision(True)
cdef (vector3d, vector3d) orthogonal_3d(vector3d vec) nogil:
    '''
    Returns two normalized vectors that are orthogonal to `vec` and orthogonal
    to eachother
    '''
    cdef:
        vector3d avec, out1, out2
        float mag

    avec.x = fabs(vec.x)
    avec.y = fabs(vec.y)
    avec.z = fabs(vec.z)
    if (avec.x <= avec.y) & (avec.x <= avec.z):
        out1.x = 0.0
        out1.y = -vec.z
        out1.z = vec.y

    elif (avec.y <= avec.x) & (avec.y <= avec.z):
        out1.x = -vec.z
        out1.y = 0.0
        out1.z = vec.x

    else:
        out1.x = -vec.y
        out1.y = vec.x
        out1.z = 0.0

    out2.x =  (vec.y*out1.z - vec.z*out1.y)
    out2.y = -(vec.x*out1.z - vec.z*out1.x)
    out2.z =  (vec.x*out1.y - vec.y*out1.x)

    mag = sqrt(out1.x**2 + out1.y**2 + out1.z**2)
    out1.x /= mag
    out1.y /= mag
    out1.z /= mag

    mag = sqrt(out2.x**2 + out2.y**2 + out2.z**2)
    out2.x /= mag
    out2.y /= mag
    out2.z /= mag
    return out1, out2


cdef vector1d transform_to_line(vector2d pnt,
                                vector2d org,
                                vector2d norm) nogil:
    '''
    Project `pnt` into a 1d coordinate system with origin `org` and the basis
    vector returned by `orthogonal_2d(norm)`.
    '''
    cdef:
        vector2d orth
        vector1d out

    orth = orthogonal_2d(norm)
    out.x = orth.x*(pnt.x - org.x) + orth.y*(pnt.y - org.y)
    return out


cdef vector2d transform_from_line(vector1d pnt,
                                  vector2d org,
                                  vector2d norm) nogil:
    '''
    Project `pnt` back into a 2d coordinate system
    '''
    cdef:
        vector2d out, orth

    orth = orthogonal_2d(norm)
    out.x = org.x + orth.x*pnt.x
    out.y = org.y + orth.y*pnt.x
    return out


@cdivision(True)
cdef vector2d transform_to_plane(vector3d pnt,
                                 vector3d org,
                                 vector3d norm) nogil:
    '''
    Project `pnt` into a 2d coordinate system with origin `org` and the basis
    vectors returned by `orthogonal_3d(norm)`.
    '''
    cdef:
        vector3d orth1, orth2
        vector2d out

    orth1, orth2 = orthogonal_3d(norm)
    out.x = (orth1.x*(pnt.x - org.x) +
             orth1.y*(pnt.y - org.y) +
             orth1.z*(pnt.z - org.z))
    out.y = (orth2.x*(pnt.x - org.x) +
             orth2.y*(pnt.y - org.y) +
             orth2.z*(pnt.z - org.z))
    return out


@cdivision(True)
cdef vector3d transform_from_plane(vector2d pnt,
                                   vector3d org,
                                   vector3d norm) nogil:
    '''
    Project `pnt` back into a 3d coordinate system
    '''
    cdef:
        vector3d out, orth1, orth2

    orth1, orth2 = orthogonal_3d(norm)
    out.x = org.x + orth1.x*pnt.x + orth2.x*pnt.y
    out.y = org.y + orth1.y*pnt.x + orth2.y*pnt.y
    out.z = org.z + orth1.z*pnt.x + orth2.z*pnt.y
    return out


@cdivision(True)
cdef bint point_in_segment(vector1d vec, segment1d seg) nogil:
    '''
    Identifies whether a point in 1D space is within a 1D segment. For the sake
    of consistency, this is done by projecting the point into a barycentric
    coordinate system defined by the segment. The point is then inside the
    segment if both of the barycentric coordinates are positive
    '''
    cdef:
        double l1, l2

    # find barycentric coordinates
    l1 = (seg.a.x - vec.x)/(seg.a.x - seg.b.x)
    l2 = (vec.x - seg.b.x)/(seg.a.x - seg.b.x)
    if (l1 >= 0.0) & (l2 >= 0.0):
        return True
    else:
        return False


@cdivision(True)
cdef bint point_in_triangle(vector2d vec, triangle2d tri) nogil:
    '''
    Identifies whether a point in 2D space is within a 2D triangle. This is
    done by projecting the point into a barycentric coordinate system defined
    by the triangle. The point is then inside the segment if all three of the
    barycentric coordinates are positive
    '''
    cdef:
        vector2d vec1, vec2, vec3
        double det, l1, l2, l3

    vec1.x = tri.a.x - tri.c.x
    vec1.y = tri.a.y - tri.c.y

    vec2.x = tri.b.x - tri.c.x
    vec2.y = tri.b.y - tri.c.y

    vec3.x = vec.x - tri.c.x
    vec3.y = vec.y - tri.c.y

    det = vec1.x*vec2.y - vec1.y*vec2.x
    l1 = ( vec2.y*vec3.x - vec2.x*vec3.y)/det
    l2 = (-vec1.y*vec3.x + vec1.x*vec3.y)/det
    l3 = 1 - l1 - l2
    if (l1 >= 0.0) & (l2 >= 0.0) & (l3 >= 0.0):
        return True
    else:
        return False


cdef vector2d segment_normal(segment2d seg) nogil:
    '''
    Returns the vector normal to a 2d line segment
    '''
    cdef:
        vector2d out

    out.x = seg.b.y - seg.a.y
    out.y = -(seg.b.x - seg.a.x)
    return out


cdef vector3d triangle_normal(triangle3d tri) nogil:
    '''
    Returns the vector normal to a 3d triangle
    '''
    cdef:
        vector3d out, vec1, vec2

    # create two vectors from the triangle, and then cross them to get the
    # normal
    vec1.x = tri.b.x - tri.a.x
    vec1.y = tri.b.y - tri.a.y
    vec1.z = tri.b.z - tri.a.z

    vec2.x = tri.c.x - tri.a.x
    vec2.y = tri.c.y - tri.a.y
    vec2.z = tri.c.z - tri.a.z

    out.x =  vec1.y*vec2.z - vec1.z*vec2.y
    out.y = -vec1.x*vec2.z + vec1.z*vec2.x
    out.z =  vec1.x*vec2.y - vec1.y*vec2.x
    return out


cdef vector2d nearest_point_in_segment(vector2d pnt, segment2d seg) nogil:
    '''
    Returns the point on `seg` that is closest to `pnt`
    '''
    cdef:
        vector2d norm, out
        vector1d pnt_proj
        segment1d seg_proj

    norm = segment_normal(seg)
    pnt_proj = transform_to_line(pnt, seg.a, norm)
    seg_proj.a = transform_to_line(seg.a, seg.a, norm)
    seg_proj.b = transform_to_line(seg.b, seg.a, norm)
    # if the point projects inside the segment, then use that projected point
    if point_in_segment(pnt_proj, seg_proj):
        out = transform_from_line(pnt_proj, seg.a, norm)

    # otherwise use the closest vertex
    elif distance_1d(pnt_proj, seg_proj.a) < distance_1d(pnt_proj, seg_proj.b):
        out = seg.a

    else:
        out = seg.b

    return out

cdef vector3d nearest_point_in_triangle(vector3d pnt, triangle3d tri) nogil:
    '''
    Returns the point on `seg` that is closest to `pnt`
    '''
    cdef:
        double dist_ab, dist_bc, dist_ca
        vector3d norm, out
        vector2d pnt_proj, nrst_ab, nrst_bc, nrst_ca
        triangle2d tri_proj
        segment2d seg_proj

    norm = triangle_normal(tri)
    pnt_proj = transform_to_plane(pnt, tri.a, norm)
    tri_proj.a = transform_to_plane(tri.a, tri.a, norm)
    tri_proj.b = transform_to_plane(tri.b, tri.a, norm)
    tri_proj.c = transform_to_plane(tri.c, tri.a, norm)
    # if the point projects inside the triangle, then use that projected point
    if point_in_triangle(pnt_proj, tri_proj):
        out = transform_from_plane(pnt_proj, tri.a, norm)

    # otherwise use closest point on the edges
    else:
        seg_proj.a = tri_proj.a
        seg_proj.b = tri_proj.b
        nrst_ab = nearest_point_in_segment(pnt_proj, seg_proj)
        dist_ab = distance_2d(pnt_proj, nrst_ab)

        seg_proj.a = tri_proj.b
        seg_proj.b = tri_proj.c
        nrst_bc = nearest_point_in_segment(pnt_proj, seg_proj)
        dist_bc = distance_2d(pnt_proj, nrst_bc)

        seg_proj.a = tri_proj.c
        seg_proj.b = tri_proj.a
        nrst_ca = nearest_point_in_segment(pnt_proj, seg_proj)
        dist_ca = distance_2d(pnt_proj, nrst_ca)

        if (dist_ab <= dist_bc) & (dist_ab <= dist_ca):
            out = transform_from_plane(nrst_ab, tri.a, norm)

        elif (dist_bc <= dist_ab) & (dist_bc <= dist_ca):
            out = transform_from_plane(nrst_bc, tri.a, norm)

        else:
            out = transform_from_plane(nrst_ca, tri.a, norm)

    return out


@cdivision(True)
cdef bint segment_intersects_segment(segment2d seg1, segment2d seg2) nogil:
    '''
    Identifies whether two 2D segments intersect. An intersection is detected
    if both segments are not colinear and if any part of the two segments touch
    '''
    cdef:
        double proj1, proj2, t
        vector2d pnt, norm
        vector1d pnt_proj
        segment1d seg_proj

    # find the normal vector components for segment 2
    norm = segment_normal(seg2)

    # project both points in segment 1 onto the normal vector
    proj1 = (seg1.a.x - seg2.a.x)*norm.x + (seg1.a.y - seg2.a.y)*norm.y
    proj2 = (seg1.b.x - seg2.a.x)*norm.x + (seg1.b.y - seg2.a.y)*norm.y

    if proj1*proj2 > 0:
        return False

    # return false if the segments are collinear
    if (proj1 == 0) & (proj2 == 0):
        return False

    # find the point where segment 1 intersects the line overlapping segment 2
    t = proj1/(proj1 - proj2)
    pnt.x = seg1.a.x + t*(seg1.b.x - seg1.a.x)
    pnt.y = seg1.a.y + t*(seg1.b.y - seg1.a.y)

    # we need to now project the segment and intersection to 1d. We could use
    # `transform_to_line`, but that involves expensive trig operations. Instead
    # we just throw out one of the components.

    # if the normal x component is larger then compare y values
    if fabs(norm.x) >= fabs(norm.y):
        pnt_proj.x = pnt.y
        seg_proj.a.x = seg2.a.y
        seg_proj.b.x = seg2.b.y
        return point_in_segment(pnt_proj, seg_proj)

    else:
        pnt_proj.x = pnt.x
        seg_proj.a.x = seg2.a.x
        seg_proj.b.x = seg2.b.x
        return point_in_segment(pnt_proj, seg_proj)


@cdivision(True)
cdef bint segment_intersects_triangle(segment3d seg, triangle3d tri) nogil:
    '''
    Identifies whether a 3D segment intersects a 3D triangle. An intersection
    is detected if the segment and triangle are not coplanar and if any part of
    the segment touches the triangle at an edge or in the interior.
    '''
    cdef:
        double proj1, proj2, t
        vector3d pnt, norm, anorm
        vector2d pnt_proj
        triangle2d tri_proj

    # find triangle normal vector components
    norm = triangle_normal(tri)

    proj1 = ((seg.a.x - tri.a.x)*norm.x +
             (seg.a.y - tri.a.y)*norm.y +
             (seg.a.z - tri.a.z)*norm.z)
    proj2 = ((seg.b.x - tri.a.x)*norm.x +
             (seg.b.y - tri.a.y)*norm.y +
             (seg.b.z - tri.a.z)*norm.z)

    if proj1*proj2 > 0:
        return False

    # coplanar segments will always return false There is a possibility that
    # the segment touches one point on the triangle
    if (proj1 == 0) & (proj2 == 0):
        return False

    # intersection point
    t = proj1/(proj1 - proj2)
    pnt.x = seg.a.x + t*(seg.b.x - seg.a.x)
    pnt.y = seg.a.y + t*(seg.b.y - seg.a.y)
    pnt.z = seg.a.z + t*(seg.b.z - seg.a.z)

    anorm.x = fabs(norm.x)
    anorm.y = fabs(norm.y)
    anorm.z = fabs(norm.z)
    if (anorm.x >= anorm.y) & (anorm.x >= anorm.z):
        pnt_proj.x = pnt.y
        pnt_proj.y = pnt.z
        tri_proj.a.x = tri.a.y
        tri_proj.a.y = tri.a.z
        tri_proj.b.x = tri.b.y
        tri_proj.b.y = tri.b.z
        tri_proj.c.x = tri.c.y
        tri_proj.c.y = tri.c.z
        return point_in_triangle(pnt_proj, tri_proj)

    elif (anorm.y >= anorm.x) & (anorm.y >= anorm.z):
        pnt_proj.x = pnt.x
        pnt_proj.y = pnt.z
        tri_proj.a.x = tri.a.x
        tri_proj.a.y = tri.a.z
        tri_proj.b.x = tri.b.x
        tri_proj.b.y = tri.b.z
        tri_proj.c.x = tri.c.x
        tri_proj.c.y = tri.c.z
        return point_in_triangle(pnt_proj, tri_proj)

    else:
        pnt_proj.x = pnt.x
        pnt_proj.y = pnt.y
        tri_proj.a.x = tri.a.x
        tri_proj.a.y = tri.a.y
        tri_proj.b.x = tri.b.x
        tri_proj.b.y = tri.b.y
        tri_proj.c.x = tri.c.x
        tri_proj.c.y = tri.c.y
        return point_in_triangle(pnt_proj, tri_proj)


@boundscheck(False)
@wraparound(False)
def _nearest_point_2d(double[:, :] pnts,
                      double[:, :] vertices,
                      long[:, :] simplices):
    '''
    Finds the point on the simplicial complex that is closest to each point in
    `pnts`. This returns the closest point and the index of the simplex which
    the closest point is on.
    '''
    cdef:
        long i, j
        long N = pnts.shape[0]
        long M = simplices.shape[0]
        double[:, :] out_pnt = np.zeros((N, 2), dtype=float, order='c')
        long[:] out_idx = np.zeros((N,), dtype=int, order='c')
        double shortest_distance
        vector2d vec1, vec2
        segment2d seg

    for i in range(N):
        vec1.x = pnts[i, 0]
        vec1.y = pnts[i, 1]
        shortest_distance = INFINITY
        for j in range(M):
            seg.a.x = vertices[simplices[j, 0], 0]
            seg.a.y = vertices[simplices[j, 0], 1]
            seg.b.x = vertices[simplices[j, 1], 0]
            seg.b.y = vertices[simplices[j, 1], 1]
            vec2 = nearest_point_in_segment(vec1, seg)
            if distance_2d(vec1, vec2) < shortest_distance:
                out_idx[i] = j
                out_pnt[i, 0] = vec2.x
                out_pnt[i, 1] = vec2.y
                shortest_distance = distance_2d(vec1, vec2)

    return np.asarray(out_pnt), np.asarray(out_idx)


@boundscheck(False)
@wraparound(False)
def _nearest_point_3d(double[:, :] pnts,
                      double[:, :] vertices,
                      long[:, :] simplices):
    '''
    Finds the point on the simplicial complex that is closest to each point in
    `pnts`. This returns the closest point and the index of the simplex which
    the closest point is on.
    '''
    cdef:
        long i, j
        long N = pnts.shape[0]
        long M = simplices.shape[0]
        double[:, :] out_pnt = np.zeros((N, 3), dtype=float, order='c')
        long[:] out_idx = np.zeros((N,), dtype=int, order='c')
        double shortest_distance
        vector3d vec1, vec2
        triangle3d tri

    for i in range(N):
        vec1.x = pnts[i, 0]
        vec1.y = pnts[i, 1]
        vec1.z = pnts[i, 2]
        shortest_distance = INFINITY
        for j in range(M):
            tri.a.x = vertices[simplices[j, 0], 0]
            tri.a.y = vertices[simplices[j, 0], 1]
            tri.a.z = vertices[simplices[j, 0], 2]
            tri.b.x = vertices[simplices[j, 1], 0]
            tri.b.y = vertices[simplices[j, 1], 1]
            tri.b.z = vertices[simplices[j, 1], 2]
            tri.c.x = vertices[simplices[j, 2], 0]
            tri.c.y = vertices[simplices[j, 2], 1]
            tri.c.z = vertices[simplices[j, 2], 2]
            vec2 = nearest_point_in_triangle(vec1, tri)
            if distance_3d(vec1, vec2) < shortest_distance:
                out_idx[i] = j
                out_pnt[i, 0] = vec2.x
                out_pnt[i, 1] = vec2.y
                out_pnt[i, 2] = vec2.z
                shortest_distance = distance_3d(vec1, vec2)

    return np.asarray(out_pnt), np.asarray(out_idx)


@boundscheck(False)
@wraparound(False)
def _intersection_count_2d(double[:, :] start_pnts,
                           double[:, :] end_pnts,
                           double[:, :] vertices,
                           long[:, :] simplices):
    '''
    Returns an array containing the number of simplices intersected between
    start_pnts and end_pnts.
    '''
    cdef:
        long i, j
        long N = start_pnts.shape[0]
        long M = simplices.shape[0]
        long[:] out = np.zeros((N,), dtype=int, order='c')
        segment2d seg1, seg2

    for i in range(N):
        seg1.a.x = start_pnts[i, 0]
        seg1.a.y = start_pnts[i, 1]
        seg1.b.x = end_pnts[i, 0]
        seg1.b.y = end_pnts[i, 1]
        for j in range(M):
            seg2.a.x = vertices[simplices[j, 0], 0]
            seg2.a.y = vertices[simplices[j, 0], 1]
            seg2.b.x = vertices[simplices[j, 1], 0]
            seg2.b.y = vertices[simplices[j, 1], 1]
            if segment_intersects_segment(seg1, seg2):
                out[i] += 1

    return np.asarray(out)


@boundscheck(False)
@wraparound(False)
def _intersection_count_3d(double[:, :] start_pnts,
                           double[:, :] end_pnts,
                           double[:, :] vertices,
                           long[:, :] simplices):
    '''
    Returns an array of the number of intersections between each line segment,
    described by start_pnts and end_pnts, and the simplices
    '''
    cdef:
        int i, j
        int N = start_pnts.shape[0]
        int M = simplices.shape[0]
        long[:] out = np.zeros((N,), dtype=int, order='c')
        segment3d seg
        triangle3d tri

    for i in range(N):
        seg.a.x = start_pnts[i, 0]
        seg.a.y = start_pnts[i, 1]
        seg.a.z = start_pnts[i, 2]
        seg.b.x = end_pnts[i, 0]
        seg.b.y = end_pnts[i, 1]
        seg.b.z = end_pnts[i, 2]
        for j in range(M):
            tri.a.x = vertices[simplices[j, 0], 0]
            tri.a.y = vertices[simplices[j, 0], 1]
            tri.a.z = vertices[simplices[j, 0], 2]
            tri.b.x = vertices[simplices[j, 1], 0]
            tri.b.y = vertices[simplices[j, 1], 1]
            tri.b.z = vertices[simplices[j, 1], 2]
            tri.c.x = vertices[simplices[j, 2], 0]
            tri.c.y = vertices[simplices[j, 2], 1]
            tri.c.z = vertices[simplices[j, 2], 2]
            if segment_intersects_triangle(seg, tri):
                out[i] += 1

    return np.asarray(out)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def _intersection_2d(double[:, :] start_pnts,
                     double[:, :] end_pnts,
                     double[:, :] vertices,
                     long[:, :] simplices):
    '''
    Returns the intersection point and the simplex being intersected by the
    segment defined by `start_pnts` and `end_pnts`.

    Notes
    -----
    if there is no intersection then a ValueError is returned. If there are
    multiple intersections, then the intersection closest to `start_pnts` will
    be returned.

    '''
    cdef:
        long i, j
        long N = start_pnts.shape[0]
        long M = simplices.shape[0]
        long[:] out_idx = np.empty((N,), dtype=int, order='c')
        double[:, :] out_pnt = np.empty((N, 2), dtype=float, order='c')
        double proj1, proj2, t, tmin
        bint found_intersection
        segment2d seg1, seg2
        vector2d norm

    for i in range(N):
        seg1.a.x = start_pnts[i, 0]
        seg1.a.y = start_pnts[i, 1]
        seg1.b.x = end_pnts[i, 0]
        seg1.b.y = end_pnts[i, 1]
        found_intersection = False
        tmin = INFINITY
        for j in range(M):
            seg2.a.x = vertices[simplices[j, 0], 0]
            seg2.a.y = vertices[simplices[j, 0], 1]
            seg2.b.x = vertices[simplices[j, 1], 0]
            seg2.b.y = vertices[simplices[j, 1], 1]
            if segment_intersects_segment(seg1, seg2):
                found_intersection = True
                # the intersecting segment should be the first segment
                # intersected when going from seg1.a to seg1.b
                norm = segment_normal(seg2)
                proj1 = ((seg1.a.x - seg2.a.x)*norm.x +
                         (seg1.a.y - seg2.a.y)*norm.y)
                proj2 = ((seg1.b.x - seg2.a.x)*norm.x +
                         (seg1.b.y - seg2.a.y)*norm.y)
                # t is a scalar between 0 and 1. If t=0 then the intersection
                # is at seg1.a and if t=1 then the intersection is at seg1.b
                t = proj1/(proj1 - proj2)
                if t < tmin:
                    tmin = t
                    out_idx[i] = j
                    out_pnt[i, 0] = seg1.a.x + t*(seg1.b.x - seg1.a.x)
                    out_pnt[i, 1] = seg1.a.y + t*(seg1.b.y - seg1.a.y)

        if not found_intersection:
            raise ValueError(
                'No intersection was found for segment [[%s, %s], [%s, %s]]' %
                (seg1.a.x, seg1.a.y, seg1.b.x, seg1.b.y))

    return np.asarray(out_pnt), np.asarray(out_idx)


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def _intersection_3d(double[:, :] start_pnts,
                     double[:, :] end_pnts,
                     double[:, :] vertices,
                     long[:, :] simplices):
    '''
    Returns the intersection point and the simplex being intersected by the
    segment defined by `start_pnts` and `end_pnts`.

    Notes
    -----
    if there is no intersection then a ValueError is returned. If there are
    multiple intersections, then the intersection closest to `start_pnts` will
    be returned.
    '''
    cdef:
        int i, j
        int N = start_pnts.shape[0]
        int M = simplices.shape[0]
        double proj1, proj2, t, tmin
        bint found_intersection
        long[:] out_idx = np.empty((N,), dtype=int, order='c')
        double[:, :] out_pnt = np.empty((N, 3), dtype=float, order='c')
        segment3d seg
        triangle3d tri
        vector3d norm

    for i in range(N):
        seg.a.x = start_pnts[i, 0]
        seg.a.y = start_pnts[i, 1]
        seg.a.z = start_pnts[i, 2]
        seg.b.x = end_pnts[i, 0]
        seg.b.y = end_pnts[i, 1]
        seg.b.z = end_pnts[i, 2]
        tmin = INFINITY
        found_intersection = False
        for j in range(M):
            tri.a.x = vertices[simplices[j, 0], 0]
            tri.a.y = vertices[simplices[j, 0], 1]
            tri.a.z = vertices[simplices[j, 0], 2]
            tri.b.x = vertices[simplices[j, 1], 0]
            tri.b.y = vertices[simplices[j, 1], 1]
            tri.b.z = vertices[simplices[j, 1], 2]
            tri.c.x = vertices[simplices[j, 2], 0]
            tri.c.y = vertices[simplices[j, 2], 1]
            tri.c.z = vertices[simplices[j, 2], 2]
            if segment_intersects_triangle(seg, tri):
                found_intersection = True
                norm = triangle_normal(tri)
                proj1 = ((seg.a.x - tri.a.x)*norm.x +
                         (seg.a.y - tri.a.y)*norm.y +
                         (seg.a.z - tri.a.z)*norm.z)
                proj2 = ((seg.b.x - tri.a.x)*norm.x +
                         (seg.b.y - tri.a.y)*norm.y +
                         (seg.b.z - tri.a.z)*norm.z)
                # t is a scalar between 0 and 1. If t=0 then the intersection
                # is at seg1.a and if t=1 then the intersection is at seg1.b
                t = proj1/(proj1 - proj2)
                if t < tmin:
                    tmin = t
                    out_idx[i] = j
                    out_pnt[i, 0] = seg.a.x + t*(seg.b.x - seg.a.x)
                    out_pnt[i, 1] = seg.a.y + t*(seg.b.y - seg.a.y)
                    out_pnt[i, 2] = seg.a.z + t*(seg.b.z - seg.a.z)

        if not found_intersection:
            raise ValueError(
                'No intersection was found for segment '
                '[[%s, %s, %s], [%s, %s, %s]]' %
                (seg.a.x, seg.a.y, seg.a.z, seg.b.x, seg.b.y, seg.b.z))

    return np.asarray(out_pnt), np.asarray(out_idx)


# end-user functions
####################################################################
def intersection_point(start_points, end_points, vertices, simplices):
    '''
    Returns the intersection between line segments and a simplicial complex.
    The line segments are described by `start_points` and `end_points`, and the
    simplicial complex is described by `vertices` and `simplices`. This
    function works for 2 and 3 spatial dimensions.

    Parameters
    ----------
    start_points : (N, D) float array
        Vertices describing one end of the line segments. `N` is the number of
        line segments and `D` is the number of dimensions

    end_points : (N, D) float array
        Vertices describing the other end of the line segments.

    vertices : (M, D) float array
        Vertices within the simplicial complex. M is the number of vertices.

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (N, D) float array
        The points where the line segments intersect the simplicial complex

    out : (N,) int array
        The index of the simplex that the line segments intersect

    Notes
    -----
    This function fails when a intersection is not found for a line segment. If
    there are multiple intersections then the intersection closest to
    start_point is used.

    '''
    start_points = np.asarray(start_points, dtype=float)
    end_points = np.asarray(end_points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=int)

    assert_shape(start_points, (None, None), 'start_points')
    assert_shape(end_points, start_points.shape, 'end_points')
    dim = start_points.shape[1]
    assert_shape(vertices, (None, dim), 'vertices')
    assert_shape(simplices, (None, dim), 'simplices')

    if dim == 2:
        out = _intersection_2d(start_points, end_points, vertices, simplices)
    elif dim == 3:
        out = _intersection_3d(start_points, end_points, vertices, simplices)
    else:
        raise ValueError('The number of spatial dimensions must be 2 or 3')

    return out


def intersection_count(start_points, end_points, vertices, simplices):
    '''
    Returns the number of simplices crossed by the line segments. The line
    segments are described by `start_points` and `end_points`. This function
    works for 2 and 3 spatial dimensions.

    Parameters
    ----------
    start_points : (N, D) array
        Vertices describing one end of the line segments. `N` is the number of
        line segments and `D` is the number of dimensions

    end_points : (N, D) array
        Vertices describing the other end of the line segments

    vertices : (M, D) array
        Vertices within the simplicial complex. `M` is the number of vertices

    simplices : (P, D) array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (N,) int array
        Intersection counts

    '''
    start_points = np.asarray(start_points, dtype=float)
    end_points = np.asarray(end_points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=int)

    assert_shape(start_points, (None, None), 'start_points')
    assert_shape(end_points, start_points.shape, 'end_points')
    dim = start_points.shape[1]
    assert_shape(vertices, (None, dim), 'vertices')
    assert_shape(simplices, (None, dim), 'simplices')

    if dim == 2:
        out = _intersection_count_2d(
            start_points, end_points, vertices, simplices)
    elif dim == 3:
        out = _intersection_count_3d(
            start_points, end_points, vertices, simplices)
    else:
        raise ValueError('The number of spatial dimensions must be 2 or 3')

    return out


def contains(points, vertices, simplices):
    '''
    Returns a boolean array identifying whether the points are contained within
    a closed simplicial complex. The simplicial complex is described by
    `vertices` and `simplices`. This function works for 2 and 3 spatial
    dimensions.

    Parameters
    ----------
    points : (N, D) array
        Test points

    vertices : (M, D) array
        Vertices of the simplicial complex

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (N,) bool array
        Indicates which test points are in the simplicial complex

    Notes
    -----
    This function does not ensure that the simplicial complex is closed. If it
    is not then bogus results will be returned.

    This function determines whether a point is contained within the simplicial
    complex by finding the number of intersections between each point and an
    arbitrary outside point.  It is possible, although rare, that this function
    will fail if the line segment intersects a simplex at an edge.

    This function does not require any particular orientation for the simplices

    '''
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=int)

    assert_shape(points, (None, None), 'points')
    dim = points.shape[1]
    assert_shape(vertices, (None, dim), 'vertices')
    assert_shape(simplices, (None, dim), 'simplices')

    # randomly generate a point that is known to be outside of the domain
    rnd = np.random.uniform(0.5, 2.0, (points.shape[1],))
    outside_point = vertices.min(axis=0) - rnd*vertices.ptp(axis=0)
    outside_point = np.repeat([outside_point], points.shape[0], axis=0)
    count = intersection_count(points, outside_point, vertices, simplices)
    out = np.array(count % 2, dtype=bool)
    return out


def nearest_point(points, vertices, simplices):
    '''
    Returns the nearest point on the simplicial complex for each point in
    `points`. This works for 2 and 3 spatial dimensions.

    Parameters
    ----------
    points : (N,D) array
        Test points

    vertices : (M,D) array
        Vertices of the simplicial complex

    simplices : (P,D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    (N, D) float array
        The nearest points on the simplicial complex

    (N,) int array
        The simplex that the nearest point is on

    '''
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=int)

    assert_shape(points, (None, None), 'points')
    dim = points.shape[1]
    assert_shape(vertices, (None, dim), 'vertices')
    assert_shape(simplices, (None, dim), 'simplices')

    if dim == 2:
        out = _nearest_point_2d(points, vertices, simplices)

    elif dim == 3:
        out = _nearest_point_3d(points, vertices, simplices)

    else:
        raise ValueError('The number of spatial dimensions must be 2 or 3')

    return out


def oriented_simplices(vertices, simplices):
    '''
    Returns simplex indices that are ordered such that each simplex normal
    vector, as defined by the right hand rule, points outward

    Parameters
    ----------
    vertices : (M, D) array
        Vertices within the simplicial complex

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (P,D) int array
        Oriented simplices

    Notes
    -----
    If one dimensional simplices are given, then the simplices are returned
    unaltered.

    This function does not ensure that the simplicial complex is closed. If it
    is not then bogus results will be returned.

    '''
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.array(simplices, dtype=int, copy=True)
    assert_shape(vertices, (None, None), 'vertices')
    dim = vertices.shape[1]
    assert_shape(simplices, (None, dim), 'simplices')

    # length scale of the domain
    scale = vertices.ptp(axis=0).max()
    dx = 1e-10*scale
    # find the normal for each simplex
    norms = simplex_normals(vertices, simplices)
    # find the centroid for each simplex
    points = np.mean(vertices[simplices], axis=1)
    # push points in the direction of the normals
    points += dx*norms
    # find which simplices are oriented such that their normals point inside
    faces_inside = contains(points, vertices, simplices)
    flip_smp = simplices[faces_inside]
    flip_smp[:, [0, 1]] = flip_smp[:, [1, 0]]
    simplices[faces_inside] = flip_smp
    return simplices


def simplex_normals(vertices, simplices):
    '''
    Returns the normal vectors for each simplex. Orientation is determined by
    the right hand rule

    Parameters
    ----------
    vertices : (M, D) array
        Vertices within the simplicial complex

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (P, D) array
        normals vectors

    Notes
    -----
    This is only defined for two and three dimensional simplices

    '''
    vertices = np.asarray(vertices, dtype=float)
    simplices = np.asarray(simplices, dtype=int)
    assert_shape(vertices, (None, None), 'vertices')
    dim = vertices.shape[1]
    assert_shape(simplices, (None, dim), 'simplices')

    M = vertices[simplices[:, 1:]] - vertices[simplices[:, [0]]]
    Msubs = [np.delete(M, i, -1) for i in range(dim)]
    out = np.linalg.det(Msubs)
    out[1::2] *= -1
    out = np.rollaxis(out, -1)
    out /= np.linalg.norm(out, axis=-1)[..., None]
    return out


def simplex_outward_normals(vertices, simplices):
    '''
    Returns the outward normal vectors for each simplex. The sign of the
    returned vectors are only meaningful if the simplices enclose an area in
    two-dimensional space or a volume in three-dimensional space

    Parameters
    ----------
    vertices : (M, D) array
        Vertices within the simplicial complex

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    Returns
    -------
    out : (P, D) array
        normals vectors

    Notes
    -----
    This is only defined for two and three dimensional simplices

    '''
    simplices = oriented_simplices(vertices, simplices)
    return simplex_normals(vertices, simplices)


def volume(vert, smp, orient=True):
    '''
    Returns the volume of a polyhedra or area of a polygon

    Parameters
    ----------
    vertices : (M, D) array
        Vertices within the simplicial complex

    simplices : (P, D) int array
        Connectivity of the vertices. Each row contains the vertex indices
        which form one simplex of the simplicial complex

    orient : bool, optional
        If true, the simplices are reordered with oriented_simplices. The time
        for this function increase quadratically with the number of simplices.
        Set to false if you are confident that the simplices are properly
        oriented. This does nothing for one-dimensional simplices

    Returns
    -------
    out : float

    Notes
    -----
    This function does not ensure that the simplicial complex is closed and
    does not intersect itself. If it is not then bogus results will be
    returned.

    '''
    vert = np.array(vert, dtype=float, copy=True)
    smp = np.asarray(smp, dtype=int)
    assert_shape(vert, (None, None), 'vert')
    dim = vert.shape[1]
    assert_shape(smp, (None, dim), 'smp')
    if orient:
        smp = oriented_simplices(vert, smp)

    # center the vertices for the purpose of numerical stability
    vert -= np.mean(vert, axis=0)
    signed_volumes = (1.0/factorial(dim))*np.linalg.det(vert[smp])
    out = np.sum(signed_volumes)
    return out
