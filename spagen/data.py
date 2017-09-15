from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


class Data(object):
    '''class for io and normalization of genotype data'''

    def __init__(self, traw, geo):

        self._get_genotype_matrix(traw)
        self._get_geographic_coordinates(geo)

    def _get_genotype_matrix(self, traw):
        '''get genotype matrix from provided traw plink file path

        Args:
            traw: str
                path to traw genotype file output from plink2
        '''
        # read and assign ids and snps
        geno_df = pd.read_table(traw, sep='\t')
        self.snps = geno_df['SNP'].as_matrix()
        self.ids = np.array(geno_df.columns)

        # get genotype matrix
        geno_df.drop(['CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT'], axis=1, inplace=True)
        self.y = geno_df.as_matrix()

    def _get_geographic_coordinates(self, geo):
        '''get geographic coordinates from clst file path

        Args:
            clst: str
                path to clst file with columns [id, lon, lat, lab]
        '''
        geo_df = pd.read_table(geo, sep='\t')
        geo_ids = geo_df['id']

        assert np.array_equal(geo_ids, self.ids)

        self.x = geo_df[['lon', 'lat']].as_matrix()
        self.lab = geo_df['lab'].as_matrix()






