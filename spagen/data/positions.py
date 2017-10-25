from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from shapely.geometry import Point, MultiPoint

class Positions(object):
    '''Class for handling geographic positions for each observation'''

    def __init__(self, geo, hull_buffer):
        '''Intialize locations

        Args:
            geo: str
                path to geography file
            hull_buffer: float
                size of buffer so points dont exsist
                at the very edge of the region i.e.
                the edge of the convex hull
        '''
        df = pd.read_table(geo, sep='\t')
        self.x = df[['lat', 'lon']].as_matrix()
        self.labels = df['lab'].as_matrix()

        self._define_region(hull_buffer)
        bounds = self.region.bounds
        self.x0_min = bounds[0]
        self.x0_max = bounds[2]
        self.x1_min = bounds[1]
        self.x1_max = bounds[3]

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
        point_list = [Point(x_i) for x_i in self.x.tolist()]
        self.points = MultiPoint(point_list)
        self.region = self.points.convex_hull.buffer(hull_buffer)

