from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np


class Data(object):
    '''Class for io and normalization of genotype data'''

    def __init__(self,
                 traw,
                 geo,
                 center=True,
                 scale=True,
                 impute=True,
                 ):
        '''Intialize data

        Args:
            traw: str
                path to traw genotype file output from plink2
            geo: str
                path to clst file with columns [id, lon, lat, lab]
            center: bool
                center the genotype matrix i.e. each snp has mean 0 accross
                individuals
            scale: bool
                scale the genotype matrix i.e. each SNP has unit variance
                individuals
            impute: bool
                fill in missing data with mean accross individuals at that SNP
        '''
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
        '''Get genotype matrix from provided traw plink file path

        Args:
            traw: str
                path to traw genotype file output from plink2
        '''
        # read traw file
        geno_df = pd.read_table(traw, sep='\t')

        # get genotype matrix
        geno_df.drop(['CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT'], axis=1, inplace=True)
        self.ids = np.array(geno_df.columns)
        self.y = geno_df.as_matrix()

    def _get_geographic_coordinates(self, geo):
        '''Get geographic coordinates from clst file path

        Args:
            geo: str
                path to clst file with columns [id, lon, lat, lab]
        '''
        geo_df = pd.read_table(geo, sep='\t')
        id_df = pd.DataFrame({'id': self.ids})
        df = id_df.merge(geo_df, how='left')

        # write assertions

        self.x = df[['lon', 'lat']].as_matrix()
        self.lab = df['lab'].as_matrix()

    def _center(self):
        '''Mean center the data matrix
        '''
        mu = np.nanmean(self.y, axis=1, keepdims=True)
        self.y = self.y - mu

    def _scale(self):
        '''Scale the data matrix to unit variance
        '''
        sigma = np.nanstd(self.y, axis=1, keepdims=True)
        self.y = self.y / sigma

    def _impute(self):
        '''Replace nans with rowmeans
        '''
        mu = np.nanmean(self.y, axis=1, keepdims=True)
        idx = np.where(np.isnan(self.y))
        self.y[idx]=np.take(mu, idx[1])

    def subsample(self, n_samp, p_samp, seed=1990):
        '''Subsample the data

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

