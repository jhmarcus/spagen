from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
import edward as ed

class FactorAnalysis(object):
    ''' '''

    def __init__(self,
                 data,
                 k=2,
                 spatial_effect=True
                 sparse_loadings=True
                 ):
        '''intialize factor analysis model

        Args:
            data: Data
                Data object from data.py
            k: int
                dimension of l_i
            spatial_effect: bool
                include spatial random effect parameterized as a gaussian process
            sparse_loadings: bool

        '''
        self.p, self.n = data.y.shape
        self.k = k
        self.spatial_effect = spatial_effect
        self.sparse_loadings = sparse_loadings
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._set_prior()
        self._set_likelihood

    def _set_prior(self)
        '''
        '''
        if self.sparse_loadings:
            # l_ij ~ N(l_ij|0, sigma2_l_ij)
            self.sigma_l = tf.maximum(tf.nn.softplus(tf.Variable(tf.random_normal([self.n, self.k]))), 1e-4)
            self.l = ed.models.Normal(loc=tf.zeros([self.n, self.k]), scale=sigma_l)
        else:
            # l_ij ~ N(l_ij|0, 1)
            self.l = ed.models.Normal(loc=tf.zeros([self.n, self.k]), scale=tf.ones([self.n, self.k]))

        # std dev of noise variance component
        self.sigma_e = tf.maximum(tf.nn.softplus(tf.Variable(tf.random_normal([]))), 1e-4)

        if self.spatial_effect:
            # lengthscale of rbf kernel
            self.alpha = tf.maximum(tf.nn.softplus(tf.Variable(tf.random_normal([]))), 1e-4)

            # std dev of spatial variance component
            self.sigma_s = tf.maximum(tf.nn.softplus(tf.Variable(tf.random_normal([]))), 1e-4)

    def _set_likelihood(self):
        '''
        '''
        # likelihood covariance matrix
        if spatial_effect:
            # placeholder for geographic coordinates
            self.x_ph = tf.placeholder(dtype=tf.float32, shape=(self.n, 2))
            self.v = (ed.rbf(x_ph, variance=tf.square(self.sigma_s), lengthscale=self.alpha) + # spatial
                      tf.matmul(self.l, self.l, transpose_b=True) + # low-rank
                      (sigma_e * tf.eye(self.n)) # noise
                     )
        else:
            # low-rank + noise
            self.v = tf.matmul(self.l, self.l, transpose_b=True) + (self.sigma_e * tf.eye(self.n))

        # likelihood
        self.y = ed.models.MultivariateNormalTriL(loc=tf.zeros([p, n]), scale_tril=tf.cholesky(v))

    def inference(self, n_iter=500, learning_rate=1e-4)
        '''
        '''
        # TODO: check if scale here is correct way to parameterize
        self.ql = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.n, self.K])), scale=self.sigma_l)

        if self.spatial_effect:
            inference = ed.KLqp({self.l: self.ql}, {self.y: self.data.y})
            inference.initialize(n_print=10, n_iter=n_iter)

        tf.global_variables_initializer().run()

        for i in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)

