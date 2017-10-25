from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from spagen.data import Genotypes


class test_genotypes_class(unittest.TestCase):

    def test_get_genotype_matrix(self):
        genotypes = Genotypes('tests/data/test.traw')

        self.assertEqual(genotypes.y.shape, (2, 157))
        self.assertEqual(genotypes.ids.shape, (157, ))

    def test_impute(self):
        genotypes_raw = Genotypes('tests/data/test.traw', normalize=False, impute=False)
        genotypes = Genotypes('tests/data/test.traw', normalize=True, impute=True)

        # get index of nan (should be only 1)
        nan_idx = np.where(np.isnan(genotypes_raw.y))

        # assert that the imputer should replace with 0 (b/c its normalized)
        self.assertAlmostEqual(genotypes.y[nan_idx][0], 0.0)

    def test_normalize(self):
        genotypes = Genotypes('tests/data/test.traw')

        mu = np.mean(genotypes.y, axis=1)
        std = np.std(genotypes.y, axis=1)

        is_mu_0 = np.allclose(mu, np.zeros(2))
        is_std_1 = np.allclose(std, np.ones(2))

        self.assertEqual(is_mu_0, True)
        self.assertEqual(is_std_1, True)

    def test_subsample_snps(self):
        genotypes = Genotypes('tests/data/test.traw', p_samp=1)

        self.assertEqual(genotypes.y.shape, (1, 157))

    def test_subsample_indiviudals(self):
        genotypes = Genotypes('tests/data/test.traw', n_samp=10)

        self.assertEqual(genotypes.y.shape, (2, 10))


