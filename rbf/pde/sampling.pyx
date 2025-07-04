'''
A module for generating points with a user specified distribution.
'''
import logging

import numpy as np

from rtree.index import Index, Property

from rbf.utils import assert_shape
from rbf.pde.halton import HaltonSequence
from rbf.pde.domain import as_domain

from libc.math cimport sqrt
from libc.stdint cimport int64_t

logger = logging.getLogger(__name__)


cdef double distance(list a, list b):
    '''
    computes the distance between two 1d tuples without the overhead of
    np.linalg.norm
    '''
    cdef:
        int64_t i
        int64_t n = len(a)
        double out = 0.0

    for i in range(n):
        out += (a[i] - b[i])**2

    return sqrt(out)        
            

def rejection_sampling(size, rho, domain, start=0, 
                       max_sample_size=1000000):
    '''
    Returns points within the boundaries defined by `vert` and `smp` and with
    density `rho`. The nodes are generated by rejection sampling.

    Parameters
    ----------
    size : int
        Number of points to return

    rho : callable
        A function that takes an (n, d) array of points and returns the density
        at those points. This should be normalized so that the density is
        between 0 and 1.

    domain : (p, d) float array and (q, d) int array
        The vertices of the domain and the connectivity of the vertices

    start : int, optional
        The starting index for the Halton sequence, which is used to propose
        new points. Setting this value is akin to setting the seed for a random
        number generator.
        
    max_sample_size : int, optional
        max number of nodes allowed in a sample for the rejection algorithm.
        This prevents excessive RAM usage

    Returns
    -------
    (size, d) float array

    '''
    logger.debug('generating nodes with rejection sampling ...')
    domain = as_domain(domain)
    # form bounding box for the domain so that a RNG can produce values that
    # mostly lie within the domain
    lb = np.min(domain.vertices, axis=0)
    ub = np.max(domain.vertices, axis=0)
    # form a Halton sequences to generate the random number between 0 and 1
    rng = HaltonSequence(1, prime_index=0)
    # form a Halton sequence to generate the test points
    pnt_rng = HaltonSequence(domain.dim, start=start, prime_index=1)
    # initiate array of points
    points = np.zeros((0, domain.dim), dtype=float)
    # node counter
    total_samples = 0
    # I use a rejection algorithm to get a sampling of points that resemble to
    # density specified by rho. The acceptance keeps track of the ratio of
    # accepted points to tested points
    acceptance = 1.0
    while points.shape[0] < size:
        # to keep most of this loop in cython and c code, the rejection
        # algorithm is done in chunks.  The number of samples in each chunk is
        # a rough estimate of the number of samples needed in order to get the
        # desired number of accepted points.
        if acceptance == 0.0:
            sample_size = max_sample_size
        else:
            # estimated number of samples needed to get `size` accepted points
            sample_size = np.ceil((size - points.shape[0])/acceptance)
            sample_size = int(sample_size)
            # dont let `sample_size` exceed `max_sample_size`
            sample_size = min(sample_size, max_sample_size)

        # In order for a test node to be accepted, `rho` evaluated at that test
        # node needs to be larger than a random number with uniform
        # distribution between 0 and 1. Here I form the test points and those
        # random numbers
        unif = rng(sample_size)[:, 0]
        test_points = pnt_rng.uniform(lb, ub, sample_size)
        # reject test points based on random value
        test_points = test_points[rho(test_points) > unif]
        # reject test points that are outside of the domain
        test_points = test_points[domain.contains(test_points)]
        # append what remains to the collection of accepted points. If there
        # are too many new points, then cut it back down so the total size is
        # `size`
        if (test_points.shape[0] + points.shape[0]) > size:
            test_points = test_points[:(size - points.shape[0])]

        points = np.vstack((points, test_points))
        logger.debug(
            'accepted %s of %s points' % (points.shape[0], size))
        # update the acceptance. the acceptance is the ratio of accepted points
        # to sampled points
        total_samples += sample_size
        acceptance = points.shape[0]/total_samples

    logger.debug('generated %s nodes with rejection sampling' 
                 % points.shape[0])
    return points


class _DiscCollection:
    '''
    A class used within `poisson_discs`. This class is a container for discs,
    where a disc is described by a center and a radius. This class provides an
    efficient method for determining whether a query disc intersects and discs
    in the collection.
    '''
    def __init__(self, dim):
        if dim == 2:
            p = Property()
            p.dimension = 2
            tree = Index(properties=p)
            
        elif dim == 3:
            p = Property()
            p.dimension = 3
            tree = Index(properties=p)
            
        else:
            raise ValueError()            

        self.tree = tree
        self.centers = []
        self.radii = []
    
    def add_disc(self, cnt, rad):
        '''
        Add a disc with center `cnt` and radius `rad` to the collection

        Parameters
        ----------
        cnt : (dim,) tuple

        rad : float
        
        '''
        lower_bounds = [c - rad for c in cnt]
        upper_bounds = [c + rad for c in cnt]
        bounds = lower_bounds + upper_bounds
        self.tree.add(len(self.centers), bounds)
        self.centers += [cnt]
        self.radii += [rad]
            
    def intersects(self, cnt, rad):
        '''
        Returns True if the disc with center `cnt` and radius `rad` overlaps
        the center of any disc in this collection OR if any of the discs in
        this collection overlap `cnt`.

        Parameters
        ----------
        cnt : (dim,) tuple

        rad : float

        '''
        lower_bounds = [c - rad for c in cnt]
        upper_bounds = [c + rad for c in cnt]
        query_bounds = lower_bounds + upper_bounds
        for idx in self.tree.intersection(query_bounds):
            dist = distance(cnt, self.centers[idx])
            if (dist < rad) | (dist < self.radii[idx]):
                return True

        return False                
        
    
def poisson_discs(rfunc, domain, seeds=10, ntests=50, 
                  rmax_factor=1.5):
    '''
    Generates Poisson disc points within the domain defined by `vert` and
    `smp`. Poisson disc points are tightly packed but are no closer than a user
    specified value. This algorithm is based on [1], and it has been modified
    to allow for spatially variable spacing. This works for two and three
    spatial dimension.

    Parameters
    ----------
    rfunc : callable
        A function that takes a (n, d) array of points as input and returns the
        desired minimum nearest neighbor distance for those points.

    domain : (p, d) float array and (q, d) int array
        The vertices of the domain and the connectivity of the vertices

    seeds : int
        The number of initial points, which are generated from a Halton
        sequence.

    ntests : int, optional
        The maximum number of attempts at finding a new neighbor for a point
        before giving up. Increasing this generally results in tighter packing.

    rmax_factor : float, optional
        
    Returns
    -------
    (n, d) float array
    
    References
    ----------
    [1] Bridson, R., Fast Poisson Disk Sampling in Arbitrary Dimensions.

    '''
    logger.debug('generating nodes with Poisson disc sampling ...')
    domain = as_domain(domain)
    # we will first generate the discs within the bounding box with lower
    # bounds `lb` and upper bounds `ub`. Nodes that are out of the bounds
    # defined by `vert` and `smp` are clipped off at the end
    lb = np.min(domain.vertices, axis=0)
    ub = np.max(domain.vertices, axis=0)
    # create the disc collection and give it some initial seed discs
    dc = _DiscCollection(domain.dim)
    centers = HaltonSequence(domain.dim).uniform(lb, ub, size=seeds)
    radii = rfunc(centers)
    for c, r in zip(centers, radii):
        dc.add_disc(c.tolist(), r)
        
    active = list(range(seeds))
    # initialize some Halton sequences as random number generators. By using
    # Halton sequences, I am ensuring that the output is deterministic without
    # messing with the global RNG seeds.
    idx_rng = HaltonSequence(1, prime_index=0)
    pnt_rng = HaltonSequence(domain.dim - 1, prime_index=1)
    while active:
        # randomly pick a disc index from `active`
        i = active[idx_rng.randint(len(active))[0]]
        center_i = dc.centers[i]
        radius_i = dc.radii[i]
        # the minimum and maximum distance that the test points can be away
        # from the chosen center.
        rmin, rmax = radius_i, rmax_factor*radius_i
        if domain.dim == 2:
            # generate test points around the selected disc. The test points
            # gradually move outward and the angle is picked from a uniform
            # distribution
            r = np.linspace(rmin, rmax, ntests + 1)[1:]
            theta, = pnt_rng.uniform([0], [2*np.pi], ntests).T
            x = center_i[0] + r*np.cos(theta)
            y = center_i[1] + r*np.sin(theta)
            # toss out test points that are out of bounds 
            keep = ((x >= lb[0]) & (x <= ub[0]) &
                    (y >= lb[1]) & (y <= ub[1]))
            cnts = np.array([x[keep], y[keep]]).T
            rads = rfunc(cnts)

        elif domain.dim == 3:
            # generate test points around the selected disc. The test points
            # gradually move outward and the angles are picked from a uniform
            # distribution
            r = np.linspace(rmin, rmax, ntests + 1)[1:]
            theta, phi = pnt_rng.uniform([      0,     0], 
                                         [2*np.pi, np.pi], 
                                         ntests).T 
            x = center_i[0] + r*np.cos(theta)*np.sin(phi)
            y = center_i[1] + r*np.sin(theta)*np.sin(phi)
            z = center_i[2] + r*np.cos(phi)
            # toss out test points that are out of bounds 
            keep = ((x >= lb[0]) & (x <= ub[0]) &
                    (y >= lb[1]) & (y <= ub[1]) &
                    (z >= lb[2]) & (z <= ub[2]))
            cnts = np.array([x[keep], y[keep], z[keep]]).T
            rads = rfunc(cnts)

        placed_disc = False
        for c, r in zip(cnts, rads):
            # test whether the test disc contains the centers of surrounding
            # discs or the surrounding discs contain the center of the test
            # disc
            if dc.intersects(c.tolist(), r):
                continue
                
            # create a new disc with center `c` and radius `r`
            dc.add_disc(c.tolist(), r)
            if (len(dc.centers) % 1000) == 0:
                logger.debug(
                    'generated %s nodes with Poisson disc sampling '
                    '...'  % len(dc.centers))

            # this new disc is active, meaning that we will search for new
            # discs to place around it
            active += [len(dc.centers) - 1]
            placed_disc = True
            break

        if not placed_disc:
            # we cannot find a disc to place around disc i, and so disc i is no
            # longer active
            active.remove(i)
            
    nodes = np.array(dc.centers)
    # throw out nodes that are outside of the domain
    logger.debug('removing nodes that are outside of the domain ...')
    nodes = nodes[domain.contains(nodes)]
    logger.debug(
        'finished generating %s nodes with Poisson disc sampling' 
        % nodes.shape[0])
    return nodes
