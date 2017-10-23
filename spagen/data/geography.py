from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import geopandas as gpd
import shapely as shp


class Geography(object):
    '''Class for handling geographic data for observations'''

    def __init__(self,
                 geo,
                 geojson=None,
                 width=None,
                 height=None):
        '''Intialize geography

        Args:
            geo: str
                path to geography file
            geojson: str
                path to geojson file
        '''
        df = pd.read_table(geo, sep='\t')
        self.x = df[['lon', 'lat']].as_matrix()
        self.labels = df['lab'].as_matrix()

"""
    def _define_region(self, geojson):
        '''Defines a study region based off a given geojson

        Assumes the geojson is at the country scale. We first
        take the union of all countries to define a region and then
        take the convex hull of the region. This convext hull is our
        study region

        Args:
            geojson: str
                path to geojson file
        '''
        countries = gpd.read_file(geojson)

        # TODO: write asssertions about being in country level
        # TODO: write assertions about data being in given region
        # TODO: automate region selection based on data

        self.region = countries['geometry'].unary_union
        self.hull = countries['geometry'].unary_union.convex_hull

    def _define_lattice(self):
        '''Define regular lattice within the convex hull

        adapted from:
        http://portolan.leaffan.net/creating-sample-points-with-ogr-and-shapely-introduction/
        '''
        ll = self.hull.bounds[:2]
        ur = self.hull.bounds[2:]
        low_x = int(ll[0]) / self.width**2
        upp_x = int(ur[0]) / self.width**2 + self.width
        low_y = int(ll[1]) / self.height**2
        upp_y = int(ur[1]) / self.height**2 + self.height

        points = []
        for x in np.linspace(low_x, upp_x, x_interval):
            for y in np.linspace(low_y, upp_y, y_interval):
                point = shp.geometry.Point(x, y)
                if point.within(self.hull):
                    points.append(point)

        self.lattice = gpd.GeoDataFrame({'geometry': points})
"""
