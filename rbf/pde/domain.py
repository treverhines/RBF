''' 
This module contains functions that generate simplices defining 
commonly used domains.
'''
import logging

import numpy as np

from rtree.index import Property, Index

from rbf.utils import assert_shape, KDTree
from rbf.pde import geometry as geo

logger = logging.getLogger(__name__)


def as_domain(obj):
    '''
    Coerces the input to a `Domain` instance. The input can be a tuple
    of vertices and simplices, or it can already be a `Domain`
    instance. In the latter case, the object will be returned
    unaltered.
    '''
    if issubclass(type(obj), Domain):
        return obj
    else:
        return Domain(*obj)
        

class Domain(object):
    '''
    A class used to facilitate computational geometry opperations on a
    domain defined by a closed collection of simplices (e.g., line
    segments or triangular facets).

    Parameters
    ----------
    vertices : (n, d) float array
        The vertices making up the domain

    simplices : (m, d) int array
        The connectivity of the vertices

    use_tree : bool, optional
        If True, then an R-Tree will be built upon initialization.
        This tree is used to speed up some of the operations.

    orient_simplices : bool, optional
        If True, then the simplices will be reoriented so that their
        normal vectors point outward.
        
    '''
    def __init__(self, vertices, simplices, 
                 use_tree=False, 
                 orient_simplices=False):
        vertices = np.asarray(vertices, dtype=float)
        simplices = np.asarray(simplices, dtype=int)
        assert_shape(vertices, (None, None), 'vertices')
        dim = vertices.shape[1]
        assert_shape(simplices, (None, dim), 'simplices')
        self.vertices = vertices
        self.simplices = simplices
        self.dim = dim     
        self.tree = None            

        if use_tree:
            # this will modify the `tree` attribute
            self._build_tree()

        if orient_simplices:
            # this will modify the `simplices` attribute. This should
            # be run after the tree has been built, otherwise it will
            # take a while
            self._orient_simplices()
            
        self.normals = geo.simplex_normals(
            self.vertices,
            self.simplices)
        
    def __repr__(self):
        return ('<Domain : '
                'vertex count=%s, '
                'simplex count=%s, '
                'using tree=%s>' % 
                (self.vertices.shape[0], 
                 self.simplices.shape[0], 
                 self.tree is not None))
                
    def _build_tree(self):
        # create a bounding box for each simplex and add those
        # bounding boxes to the R-Tree
        logger.debug('building R-Tree ...')
        smp_min = self.vertices[self.simplices].min(axis=1)
        smp_max = self.vertices[self.simplices].max(axis=1)
        bounds = np.hstack((smp_min, smp_max))
        
        p = Property()
        p.dimension = self.dim
        self.tree = Index(properties=p)
        for i, bnd in enumerate(bounds):
            self.tree.add(i, bnd)
            
        logger.debug('done')

    def _orient_simplices(self):
        logger.debug('orienting simplices ...')
        # length scale of the domain
        scale = self.vertices.ptp(axis=0).max()
        dx = 1e-10*scale
        # find the normal for each simplex
        norms = geo.simplex_normals(self.vertices, self.simplices)
        # find the centroid for each simplex
        points = np.mean(self.vertices[self.simplices], axis=1)
        # push points in the direction of the normals
        points += dx*norms
        # find which simplices are oriented such that their normals
        # point inside
        faces_inside = self.contains(points)
        # make a copy of simplices because we are modifying it in
        # place
        new_smp = np.array(self.simplices, copy=True)
        # flip the order of the simplices that are backwards
        flip_smp = new_smp[faces_inside]
        flip_smp[:, [0, 1]] = flip_smp[:, [1, 0]]
        new_smp[faces_inside] = flip_smp
        self.simplices = new_smp
        logger.debug('done')

    def intersection_count(self, start_points, end_points):
        '''
        Counts the number times the line segments intersect the
        boundary.

        Parameters
        ----------
        start_points, end_points : (n, d) float array
            The ends of the line segments

        Returns
        -------
        (n,) int array
            The number of boundary intersection

        '''
        start_points = np.asarray(start_points, dtype=float)
        end_points = np.asarray(end_points, dtype=float)
        assert_shape(start_points, (None, self.dim), 'start_points')
        assert_shape(end_points, start_points.shape, 'end_points')
        n = start_points.shape[0]
        
        if self.tree is None:
            return geo.intersection_count(
                start_points,
                end_points,
                self.vertices,
                self.simplices)

        else:
            out = np.zeros(n, dtype=int)
            # get the bounding boxes around each segment
            bounds = np.hstack((np.minimum(start_points, end_points),
                                np.maximum(start_points, end_points)))   
            for i, bnd in enumerate(bounds):
                # get a list of simplices which could potentially be
                # intersected by segment i
                potential_smpid = list(self.tree.intersection(bnd))
                if not potential_smpid:
                    # if the segment bounding box does not intersect
                    # and simplex bounding boxes, then there is no
                    # intersection
                    continue
                
                out[[i]] = geo.intersection_count(
                    start_points[[i]],
                    end_points[[i]],
                    self.vertices,
                    self.simplices[potential_smpid])

            return out                    
                    
    def intersection_point(self, start_points, end_points):
        '''
        Finds the point on the boundary intersected by the line
        segments. A `ValueError` is raised if no intersection is
        found.

        Parameters
        ----------
        start_points, end_points : (n, d) float array
            The ends of the line segments

        Returns
        -------
        (n, d) float array
            The intersection point
            
        (n,) int array
            The simplex containing the intersection point

        '''        
        # dont bother using the tree for this one
        return geo.intersection_point(
            start_points, 
            end_points,        
            self.vertices,
            self.simplices)

    def contains(self, points):
        '''
        Identifies whether the points are within the domain

        Parameters
        ----------
        points : (n, d) float array

        Returns
        -------
        (n,) bool array
        
        '''
        points = np.asarray(points, dtype=float)
        assert_shape(points, (None, self.dim), 'points')
        # to find out if the points are inside the domain, we create
        # another set of points which are definitively outside the
        # domain, and then we count the number of boundary
        # intersections between `points` and the new points.

        # get the min value and width of the domain along axis 0
        xwidth = self.vertices[:, 0].ptp()
        xmin = self.vertices[:, 0].min()
        # the outside points are directly to the left of `points` plus
        # a small random perturbation. The subsequent bounding boxes
        # are going to be very narrow, meaning that the R-Tree will
        # efficiently winnow down the potential intersecting
        # simplices.
        outside_points = np.array(points, copy=True)
        outside_points[:, 0] = xmin - xwidth
        outside_points += np.random.uniform(
            -0.001*xwidth, 
            0.001*xwidth,
            points.shape)
        count = self.intersection_count(points, outside_points)            
        # If the segment intersects the boundary an odd number of
        # times, then the point is inside the domain, otherwise it is
        # outside
        out = np.array(count % 2, dtype=bool)
        return out

    def snap(self, points, delta=0.5):
        '''
        Snaps `points` to the nearest points on the boundary if they
        are sufficiently close to the boundary. A point is
        sufficiently close if the distance to the boundary is less
        than `delta` times the distance to its nearest neighbor.

        Parameters
        ----------
        points : (n, d) float array

        delta : float, optional

        Returns
        -------
        (n, d) float array
            The new points after snapping to the boundary

        (n,) int array
            The simplex that the points are snapped to. If a point is
            not snapped to the boundary then its corresponding value
            will be -1.
        
        '''
        points = np.asarray(points, dtype=float)
        assert_shape(points, (None, self.dim), 'points')
        n = points.shape[0]

        out_smpid = np.full(n, -1, dtype=int)
        out_points = np.array(points, copy=True)
        nbr_dist = KDTree(points).query(points, 2)[0][:, 1]
        snap_dist = delta*nbr_dist

        if self.tree is None:
            nrst_pnt, nrst_smpid = geo.nearest_point(
                points,
                self.vertices,
                self.simplices)
            nrst_dist = np.linalg.norm(nrst_pnt - points, axis=1)
            snap = nrst_dist < snap_dist
            out_points[snap] = nrst_pnt[snap]
            out_smpid[snap] = nrst_smpid[snap]

        else:
            # creating bounding boxes around the snapping regions for
            # each point
            bounds = np.hstack((points - snap_dist[:, None],
                                points + snap_dist[:, None]))
            for i, bnd in enumerate(bounds):
                # get a list of simplices which node i could
                # potentially snap to
                potential_smpid = list(self.tree.intersection(bnd))
                # sort the list to ensure consistent output
                potential_smpid.sort()
                if not potential_smpid: 
                    # no simplices are within the snapping distance
                    continue
                
                # get the nearest point to the potential simplices and
                # the simplex containing the nearest point
                nrst_pnt, nrst_smpid = geo.nearest_point(
                    points[[i]],
                    self.vertices,
                    self.simplices[potential_smpid])
                nrst_dist = np.linalg.norm(points[i] - nrst_pnt[0])
                # if the nearest point is within the snapping distance
                # then snap
                if nrst_dist < snap_dist[i]:
                    out_points[i] = nrst_pnt[0]
                    out_smpid[i] = potential_smpid[nrst_smpid[0]]

        return out_points, out_smpid
    
        
def _circle_refine(vert, smp):
    V = vert.shape[0]
    S = smp.shape[0]
    new_vert = np.zeros((V+S, 2), dtype=float)
    new_vert[:V, :] = vert
    new_smp = np.zeros((2*S, 2), dtype=int)
    for si, s in enumerate(smp):
        a, b = vert[s]
        i = V + si
        new_vert[i] = a+b
        new_smp[2*si]   = [s[0],    i]
        new_smp[2*si+1] = [   i, s[1]]

    new_vert = new_vert / np.linalg.norm(new_vert, axis=1)[:, None]
    return new_vert, new_smp


def circle(r=5):
    ''' 
    Returns the outwardly oriented simplices of a circle

    Parameters
    ----------
    r : int, optional
    refinement order
      
    Returns
    -------
    vert : (N, 2) float array
    smp : (M, 2) int array

    '''
    vert = np.array([[ 1.0, 0.0],
                     [ 0.0, 1.0],
                     [-1.0, 0.0],
                     [0.0, -1.0]])
    smp = np.array([[0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0]])
    for i in range(r):
        vert, smp = _circle_refine(vert, smp)

    return vert, smp


def _sphere_refine(vert, smp):
    V = vert.shape[0]
    S = smp.shape[0]
    new_vert = np.zeros((V+3*S, 3), dtype=float)
    new_vert[:V, :] = vert
    new_smp = np.zeros((4*S, 3), dtype=int)
    for si, s in enumerate(smp):
        a, b, c = vert[s]
        i = V + 3*si
        j = i + 1
        k = i + 2
        new_vert[i] = a+b
        new_vert[j] = b+c
        new_vert[k] = a+c
        new_smp[4*si]   = [   i,    j,    k]
        new_smp[4*si+1] = [s[0],    i,    k]
        new_smp[4*si+2] = [   i, s[1],    j]
        new_smp[4*si+3] = [   k,    j, s[2]]

    new_vert = new_vert / np.linalg.norm(new_vert, axis=1)[:, None]
    return new_vert, new_smp


def sphere(r=5):
    ''' 
    Returns the outwardly oriented simplices of a sphere

    Parameters
    ----------
    r : int, optional
    refinement order
      
    Returns
    -------
    vert : (N,2) float array

    smp : (M,2) int array

    '''
    f = np.sqrt(2.0)/2.0
    vert = np.array([[ 0.0, -1.0, 0.0],
                     [  -f,  0.0,   f],
                     [   f,  0.0,   f],
                     [   f,  0.0,  -f],
                     [  -f,  0.0,  -f],
                     [ 0.0,  1.0, 0.0]])
    smp = np.array([[0, 2, 1],
                    [0, 3, 2],
                    [0, 4, 3],
                    [0, 1, 4],
                    [5, 1, 2],
                    [5, 2, 3],
                    [5, 3, 4],
                    [5, 4, 1]])

    for i in range(r):
        vert, smp = _sphere_refine(vert, smp)

    return vert, smp


def square():
    '''
    Return the simplices for a unit square
    '''
    vert = np.array([[0.0, 0.0],
                     [1.0, 0.0],
                     [1.0, 1.0],
                     [0.0, 1.0]])
    smp = np.array([[0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0]])
    return vert, smp


def cube():
    '''
    Returns the simplices for a unit cube
    '''
    vert = np.array([[0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 1.0, 1.0],
                     [1.0, 0.0, 0.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0],
                     [1.0, 1.0, 1.0]])
    smp = np.array([[1, 0, 4],
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
                    [6, 2, 7]])
    return vert, smp
