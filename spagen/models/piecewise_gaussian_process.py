from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy import stats
from shapely.geometry import Point, MultiPoint, MultiPolygon, Polygon
from scipy.spatial import distance as dist
from scipy.spatial import Voronoi


class PieceWiseGaussianProcess(object):
    '''Non-stationary gaussian process model to visualize covariance patterns over space'''

    def __init__(self, dataset, r=10, u=.33333):
        '''Intialize non-stationary gaussian process model

        Inspired by Petkova et al. 2016 and Kim et al. 2005 we
        implement a non-stationary gaussian process model to visualize
        areas of restricted migration over geographic space inferred
        using a voronoi tiling scheme / birth-death MCMC.

        Args:
            dataset: Dataset
                dataset object
            r: int
                number of sucesses parameter of negative-binomial prior
                on the number of tiles
            u: float
                probability of success paramter of negative-binomial
                prior on the number of tiles
        '''
        # data
        self.dataset = dataset
        self.x = self.dataset.positions.x
        self.y = self.dataset.genotypes.y
        self.x0_min = self.dataset.positions.x0_min
        self.x0_max = self.dataset.positions.x0_max
        self.x1_min = self.dataset.positions.x1_min
        self.x1_max = self.dataset.positions.x1_max
        self.region = self.dataset.positions.region

        # fixed prior hyper-parameters
        self.r = r
        self.u = u

    def _voronoi_finite_polygons_2d(self, vor, radius):
        '''
        reconstruct infinite voronoi regions in a 2d diagram to finite
        regions.

        parameters
        ----------
        vor : voronoi
            input diagram
        radius : float, optional
            distance to 'points at infinity'.

        returns
        -------
        regions : list of tuples
            indices of vertices in each revised voronoi regions.
        vertices : list of tuples
            coordinates for revised voronoi vertices. same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        modified from http://stackoverflow.com/a/20678647/416626
        '''

        if vor.points.shape[1] != 2:
            raise valueerror('requires 2d input')

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def _voronoi_tesselation(self, s):
        '''Perform voronoi tesselation given seed and sample locations

        Args:
            s: np.array
                cell positions
        '''
        vor = Voronoi(s)

        # clip vor tess to have large finite radius
        vor_regions, vor_vertices = self._voronoi_finite_polygons_2d(vor, radius=5000)

        # clip finite vor to convex hull
        cells = []
        for vor_reg in vor_regions:
            cells.append(Polygon(vor_vertices[vor_reg].tolist()).intersection(self.region))

        self.t = MultiPolygon(cells)
        # distance of each position to seed
        #seed_dist = dist.cdist(self.x, s, metric='euclidean')
        # assign each sample to closest seed
        #self.t = np.argmin(seed_dist, axis=1)

    def _sample_seeds(self, c):
        '''
        '''
        m = 0
        seeds = []
        while m <= c:
            x0 = npr.uniform(self.x0_min, self.x0_max, 1)[0]
            x1 = npr.uniform(self.x1_min, self.x1_max, 1)[0]
            point = Point([x0, x1])
            if self.region.contains(point):
                seeds.append([x0, x1])
                m+=1
        s = np.array(seeds)
        return(s)

    def intialize(self):
        '''Intialize paramters for mcmc
        '''
        self.c = npr.negative_binomial(self.r, self.u, 1)[0] + 2
        self.s = self._sample_seeds(self.c)


    def fit(self, n_iter, n_burn=2e4, n_thin=50):
        '''Run mcmc to estimate covariance parameters

        Args:
            n_iter: int
                number of mcmc iterations
            n_burn: int
                number of burnin iterations
            n_thin: int
                sample every n_thin iterations
        '''
        pass


