from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from spagen.data.genotypes import Genotypes
from spagen.data.geography import Geography


class Dataset(object):
    '''Dataset class consisting of genotypes and geographic information'''

    def __init__(self,
                 traw,
                 geo,
                 normalize=True,
                 impute=True,
                 n_samp=None,
                 p_samp=None,
                 geojson=None):
        '''
        Intialize the dataset

        Args:
            traw: str
                path to traw genotype file output from plink2
            geo: str
                path to geography file
            normalize: bool
                center and scale the genotype matrix i.e. each snp has mean 0 accross
                individuals and std 1.
            impute: bool
                fill in missing data with mean accross individuals at that SNP
            n_samp: int
                number of individuals to subsample
            p_samp: int
                number of snps to subsample
            geojson: str
                path to geojson file
        '''
        self.genotypes = Genotypes(traw, normalize, impute, n_samp, p_samp)
        self.geography = Geography(geo)

        # re-index geographic data if individuals are subsampled
        if n_samp != None:
            self.geography.x = self.geography.x[self.genotypes.j,:]
            self.geography.labels = self.geography.labels[self.genotypes.j]

