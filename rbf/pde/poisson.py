'''
A module for generating Poisson discs with variable disc radius
'''
import numpy as np
from rbf.utils import assert_shape
from rbf.pde.halton import halton, Halton
from scipy.spatial import cKDTree

class _DiscCollection:
    '''
    A container for discs, where a disc is described by a center and a
    radius, which also contains efficient querying methods.
    '''
    def __init__(self, centers, radii, leafsize=100):
        self.centers = centers
        self.radii = radii
        self.leafsize = leafsize
        self.tree = cKDTree(self.centers, 
                            leafsize=self.leafsize, 
                            compact_nodes=True, 
                            balanced_tree=True)
    
    def add_disc(self, cnt, rad):
        '''
        Add a disc with center `cnt` and radius `rad` to the
        collection
        '''
        self.centers = np.vstack((self.centers, [cnt]))
        self.radii = np.hstack((self.radii, [rad]))
        # only rebuild the tree after every `leafsize` new points have
        # been added.
        if (len(self.centers) % self.leafsize) == 0:
            self.tree = cKDTree(self.centers, 
                                leafsize=self.leafsize, 
                                compact_nodes=True, 
                                balanced_tree=True)
            
    def centers_in_disc(self, cnt, rad):
        '''
        Returns the indices of discs whose centers are contained in
        the disc with center `cnt` and radius `rad`
        '''        
        # use brute force to test discs that were added since the last
        # tree build
        recent_centers = self.centers[self.tree.n:]
        dist = np.linalg.norm(recent_centers - cnt[None, :], axis=1)
        out, = (dist < rad).nonzero()
        out = (out + self.tree.n).tolist()
        # use the tree to test the remaining discs
        out += self.tree.query_ball_point(cnt, rad)
        return out

    def any_centers_in_disc(self, cnt, rad):
        '''
        Returns True if any of the disc centers are contained in the
        disc with center `cnt` and radius `rad`
        '''
        # use brute force to test discs that were added since the last
        # tree build
        recent_centers = self.centers[self.tree.n:]
        dist = np.linalg.norm(recent_centers - cnt[None, :], axis=1)
        if np.any(dist < rad):
            return True
                    
        # use the tree to test the remaining discs
        elif self.tree.query_ball_point(cnt, rad):
            return True

        else:
            return False            

    def any_discs_contain_point(self, x):
        '''
        Returns True if any of the discs contain the point `x`
        '''
        rmax = np.max(self.radii)
        while True:
            # find the discs with centers that are within a distance
            # of `rmax` to `x`
            indices = self.centers_in_disc(x, rmax)
            # if no disc centers are within a distance of `rmax` to
            # `x`, then no discs contain `x`
            if not indices:
                return False

            # Find the largest radius of all the discs with centers
            # within `rmax`. 
            new_rmax = self.radii[indices].max()
            # If the largest disc radius is equal to `rmax` then it
            # must contain `x`
            if new_rmax == rmax:
                return True

            rmax = new_rmax                
        
    
def poisson_discs(rfunc, bounds, seeds=10, k=50):
    '''
    Generates poisson disc points, where the points are tightly packed
    but are no closer than a user specified value. This algorithm is
    based on [1], and it has been modified to allow for spatially
    variable spacing. This works for two and three spatial dimension.

    Parameters
    ----------
    rfunc : callable
        A function that takes a (n, d) array of points as input and
        returns the minimum nearest neighbor distance for those
        points.

    bounds : (d, 2) array
        The lower and upper bounds for each spatial dimension         

    seed : int
        The number of initial points, which are generated from a
        Halton sequence.

    k : int
        The maximum number of attempts at finding a new neighbor for a
        point before giving up.
        
    Returns
    -------
    (n, d) float array
    
    References
    ----------
    [1] Bridson, R., Fast Poisson Disk Sampling in Arbitrary
        Dimensions.

    '''
    bounds = np.asarray(bounds)
    assert_shape(bounds, (None, 2), 'bounds')
    dim = bounds.shape[0]
    
    centers = halton(seeds, dim)
    centers  = centers*bounds.ptp(axis=1) + bounds[:, 0]
    radii = np.asarray(rfunc(centers))
    dc = _DiscCollection(centers, radii)
    active = np.arange(seeds).tolist()

    # initialize some Halton sequences as random number generators. By
    # using Halton sequences, I am ensuring that the output is
    # deterministic without messing with the global RNG seeds.
    idx_rng = Halton(1, prime_index=0)
    r_rng = Halton(1, prime_index=1)
    theta_rng = Halton(1, prime_index=2)
    phi_rng = Halton(1, prime_index=3)

    while active:
        # randomly pick a disc index from `active`
        i = active[int(idx_rng(1)[0, 0]*len(active))]
        center_i = dc.centers[i]
        radius_i = dc.radii[i]
        rmin, rmax = radius_i, 2*radius_i
        if dim == 2:
            # randomly generate test points around disc i
            r = r_rng(k)[:, 0]*(rmax - rmin) + rmin
            theta = theta_rng(k)[:, 0]*2*np.pi
            x = center_i[0] + r*np.cos(theta)
            y = center_i[1] + r*np.sin(theta)
            # toss out test points that are out of bounds 
            keep = ((x >= bounds[0, 0]) & (x <= bounds[0, 1]) &
                    (y >= bounds[1, 0]) & (y <= bounds[1, 1]))
            # the centers and radii for k test discs
            cnts = np.array([x[keep], y[keep]]).T
            rads = np.asarray(rfunc(cnts))

        elif dim == 3:
            # randomly generate points around disc i
            r = r_rng(k)[:, 0]*(rmax - rmin) + rmin
            theta = theta_rng(k)[:, 0]*2*np.pi
            phi = phi_rng(k)[:, 0]*np.pi
            x = center_i[0] + r*np.cos(theta)*np.sin(phi)
            y = center_i[1] + r*np.sin(theta)*np.sin(phi)
            z = center_i[2] + r*np.cos(phi)
            # toss out test points that are out of bounds 
            keep = ((x >= bounds[0, 0]) & (x <= bounds[0, 1]) &
                    (y >= bounds[1, 0]) & (y <= bounds[1, 1]) &
                    (z >= bounds[2, 0]) & (z <= bounds[2, 1]))
            # the centers and radii for k test discs
            cnts = np.array([x[keep], y[keep], z[keep]]).T
            rads = np.asarray(rfunc(cnts))

        placed_disc = False
        for c, r in zip(cnts, rads):
            # test whether the test disc contains the centers of
            # surrounding discs
            if not dc.any_centers_in_disc(c, r):
                # test whether the surrounding discs contain the
                # center of the test disc
                if not dc.any_discs_contain_point(c):
                    # create a new disc with center `c` and radius `r`
                    dc.add_disc(c, r)
                    # this new disc is active, meaning that we will
                    # search for new discs to place around it
                    active += [len(dc.centers) - 1]
                    placed_disc = True
                    break

        if not placed_disc:
            # we cannot find a disc to place around disc i, and so
            # disc i is no longer active
            active.remove(i)
            
    return np.array(dc.centers)
