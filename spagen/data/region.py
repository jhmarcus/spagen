from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shapely.geometry import Point, MultiPoint

from spagen.data import Locations


class Region(Locations):
    '''Class for defining geographical region to model'''

    def __init__(self, geo, hull_buffer):
        '''
        Inherits geographic coordinates / args from Locations

        Args:
            geo: str
                path to geo file
            hull_buffer: float
                size of buffer so points dont exsist
                at the very edge of the region i.e.
                the edge of the convex hull
        '''
        super(Region, self).__init__(geo)
        self._define_region(hull_buffer)
        bounds = self.region.bounds
        self.x_min = bounds[0]
        self.y_min = bounds[1]
        self.x_max = bounds[2]
        self.y_max = bounds[3]

    def _define_region(self, hull_buffer):
        '''Defines a study region based off locations

        Takes convex hull (with some buffer) around a set
        given coordinates which defines the study region
        which the model will estimates parameters in

        Args:
            hull_buffer: float
                size of buffer so points dont exsist
                at the very edge of the region i.e.
                the edge of the convex hull
        '''
        point_list = [Point([x_i[1], x_i[0]]) for x_i in self.x.tolist()]
        self.points = MultiPoint(point_list)
        self.region = self.points.convex_hull.buffer(hull_buffer)


