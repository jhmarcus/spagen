from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

class Locations(object):
    '''Class for handling geographic locations and labels for observations'''

    def __init__(self, geo):
        '''Intialize locations

        Args:
            geo: str
                path to geography file
        '''
        df = pd.read_table(geo, sep='\t')
        self.x = df[['lon', 'lat']].as_matrix()
        self.labels = df['lab'].as_matrix()
