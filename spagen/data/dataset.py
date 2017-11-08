from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from spagen.data import Genotypes, Positions


class Dataset(object):
    '''Dataset class consisting of genotypes and geographic information'''

    def __init__(self,
                 traw,
                 geo,
                 normalize=True,
                 impute=True,
                 n_samp=None,
                 p_samp=None,
                 hull_buffer=2):
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
            hull_buffer: float
                size of buffer so points dont exsist
                at the very edge of the region i.e.
                the edge of the convex hull
        '''
        self.genotypes = Genotypes(traw, normalize, impute, n_samp, p_samp)
        self.positions = Positions(geo, hull_buffer)

        if n_samp != None:
            # re-index individuals
            self.positions.x = self.positions.x[self.genotypes.j,:]

            # re-index labels
            self.positions.labels = self.positions.labels[self.genotypes.j]

            # re-define region
            self.positions._define_region(self, hull_buffer)


