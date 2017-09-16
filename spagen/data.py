from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


class Data(object):
    '''class for io and normalization of genotype data'''

    def __init__(self,
                 traw,
                 geo,
                 center=True,
                 scale=True,
                 impute=True
                 ):

        self._get_genotype_matrix(traw)
        self._get_geographic_coordinates(geo)

        # dimensions
        self.p, self.n = self.y.shape

        # center and scale
        if center == True and scale ==True:
            self.centered = True
            self.scaled = True
            self._center()
            self._scale()
        elif center == True and scale == False:
            self.centered = True
            self.scaled = False
            self._center()
        elif center == False and scale == True:
            # throw error
            pass
        else:
            self.centered = False
            self.scaled = False

        # impute
        if impute:
            self.imputed = True
            self._impute()

    def _get_genotype_matrix(self, traw):
        '''get genotype matrix from provided traw plink file path

        Args:
            traw: str
                path to traw genotype file output from plink2
        '''
        # read and assign ids and snps
        geno_df = pd.read_table(traw, sep='\t')
        self.snps = geno_df['SNP'].as_matrix()

        # get genotype matrix
        geno_df.drop(['CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT'], axis=1, inplace=True)
        self.ids = np.array(geno_df.columns)
        self.y = geno_df.as_matrix()

    def _get_geographic_coordinates(self, geo):
        '''get geographic coordinates from clst file path

        Args:
            clst: str
                path to clst file with columns [id, lon, lat, lab]
        '''
        geo_df = pd.read_table(geo, sep='\t')
        id_df = pd.DataFrame({'id': self.ids})
        df = id_df.merge(geo_df, how='left')

        # write assertions

        self.x = df[['lon', 'lat']].as_matrix()
        self.lab = df['lab'].as_matrix()

    def _center(self):
        '''mean center the data matrix
        '''
        mu = np.nanmean(self.y, axis=1, keepdims=True)
        self.y = self.y - mu

    def _scale(self):
        '''scale the data matrix to unit variance
        '''
        sigma = np.nanstd(self.y, axis=1, keepdims=True)
        self.y = self.y / sigma

    def _impute(self):
        '''replace nans with rowmeans
        '''
        mu = np.nanmean(self.y, axis=1, keepdims=True)
        idx = np.where(np.isnan(self.y))
        self.y[idx]=np.take(mu, idx[1])

    def subsample(self, n_samp, p_samp, seed=1990):
        '''subsample the data

        Args:
            p_samp: int
                number of snps to sample
            n_samp: int
                number of individuals to sample
            seed: int
                seed for rng
        '''
        np.random.seed(seed=seed)

        # sample rows
        i = np.random.randint(0, self.y.shape[0], p_samp)
        self.y = self.y[i,:]
        self.p = p_samp

        # sample columns
        j = np.random.randint(0, self.y.shape[1], n_samp)
        self.y = self.y[:,j]
        self.n = n_samp

        # re-index other data
        self.x = self.x[j,:]
        self.ids = self.ids[j]
        self.lab = self.lab[j]

