from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import Imputer


class Genotypes(object):
    '''Class for io and normalization of genotype data'''

    def __init__(self,
                 traw,
                 normalize=True,
                 impute=True,
                 n_samp=None,
                 p_samp=None
                 ):
        '''Intialize data

        Args:
            traw: str
                path to traw genotype file output from plink2
            normalize: bool
                center and scale the genotype matrix i.e. each snp has mean 0 accross
                individuals and std 1
            impute: bool
                fill in missing data with mean accross individuals at that SNP
            n_samp: int
                number of individuals to subsample
            p_samp: int
                number of snps to subsample
        '''
        # read genotype matrix
        self._get_genotype_matrix(traw)
        self.p, self.n = self.y.shape

        # subsample the data
        if n_samp != None and p_samp != None:
            self._subsample_snps(p_samp)
            self._subsample_individuals(n_samp)
        elif n_samp != None and p_samp == None:
            self._subsample_individuals(n_samp)
        elif n_samp == None and p_samp != None:
            self._subsample_snps(p_samp)

        # impute
        if impute:
            self.imputed = True
            self._impute()

        # normalize
        if normalize:
            self.normalized = True
            self._normalize()

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

    def _normalize(self):
        '''Mean center and scale the data matrix
        '''
        #mu = np.nanmean(self.y, axis=1, keepdims=True)
        #std = np.nanstd(self.y, axis=1, keepdims=True)
        self.y = scale(self.y, axis=1)

    def _impute(self):
        '''Replace nans with rowmeans
        '''
        imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
        imp.fit(self.y)
        self.y = imp.transform(self.y)

    def _subsample_snps(self, p_samp):
        '''Subsample the features

        Args:
            p_samp: int
                number of snps to subsample
        '''
        self.i = np.random.randint(0, self.p, p_samp)
        self.y = self.y[self.i,:]
        self.p = p_samp

    def _subsample_individuals(self, n_samp):
        '''Subsample the samples

        Args:
            n_samp: int
                number of individuals to subsample
        '''
        self.j = np.random.randint(0, self.n, n_samp)
        self.y = self.y[:,self.j]
        self.n = n_samp
        self.ids = self.ids[self.j]

