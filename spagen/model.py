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
                 spatial_effect=True,
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
                assume sparse prior on the loadings
        '''
        self.data = data
        self.p, self.n = data.y.shape
        self.k = k
        self.spatial_effect = spatial_effect
        self.sparse_loadings = sparse_loadings

    def _prior(self):
        '''sets prior distributions for posterior inference and variables for maximum likelihood inference
        '''
        if self.sparse_loadings:
            # l_ij ~ N(l_ij|0, sigma2_ij)
            self.sigma_l = tf.nn.softplus(tf.Variable(tf.random_normal([self.n, self.k])))
            self.l = ed.models.Normal(loc=tf.zeros([self.n, self.k]), scale=self.sigma_l)
        else:
            # l_ij ~ N(l_ij|0, 1)
            self.l = ed.models.Normal(loc=tf.zeros([self.n, self.k]), scale=tf.ones([self.n, self.k]))

        # std dev of noise variance component
        self.sigma_e = tf.nn.softplus(tf.Variable(tf.random_normal([])))

        if self.spatial_effect:
            # lengthscale of rbf kernel
            self.alpha = tf.nn.softplus(tf.Variable(tf.random_normal([])))

            # std dev of spatial variance component
            self.sigma_s = tf.nn.softplus(tf.Variable(tf.random_normal([])))

    def _likelihood(self):
        '''sets likelihood where the data is the genotype matrix
        '''
        if self.spatial_effect:

            # placeholder for geographic coordinates
            self.x_ph = tf.placeholder(dtype=tf.float32, shape=(self.n, 2))

            # likelihood covariance matrix
            self.v = (ed.rbf(self.x_ph, variance=tf.square(self.sigma_s), lengthscale=self.alpha) + # spatial
                      tf.matmul(self.l, self.l, transpose_b=True) + # low-rank
                      (self.sigma_e * tf.eye(self.n)) # noise
                     )
        else:
            # likelihood covariance matrix (no spatial effect)
            self.v = tf.matmul(self.l, self.l, transpose_b=True) + (self.sigma_e * tf.eye(self.n))

        # likelihood
        self.y = ed.models.MultivariateNormalTriL(loc=tf.zeros([self.p, self.n]), scale_tril=tf.cholesky(self.v))

    def _inference(self, n_iter=10, learning_rate=1e-4):
        '''run variational em

        TODO:
            * add convergence criteria based on heldout data?
            * check ql scale is correct way to parameterize
            * add attributes that store loss function

        Args:
            n_iter: int
                number of epochs of variational em
            learning_rate: float
                learning rate for stochastic graident descent of ELBO
        '''
        self.ql = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.n, self.k])), scale=self.sigma_l)

        if self.spatial_effect:
            inference = ed.KLqp({self.l: self.ql}, {self.y: self.data.y, self.x_ph: self.data.x})
            inference.initialize(n_print=10, n_iter=n_iter)
        else:
            inference = ed.KLqp({self.l: self.ql}, {self.y: self.data.y})
            inference.initialize(n_print=10, n_iter=n_iter)

        tf.global_variables_initializer().run()

        for i in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)

    def fit(self, n_iter=500, learning_rate=1e-4):
        '''public method to peform inference on defined model

        Args:
            n_iter: int
                number of epochs of variational em
            learning_rate: float
                learning rate for stochastic VI
        '''
        self._prior()
        self._likelihood()
        self._inference(n_iter, learning_rate)

