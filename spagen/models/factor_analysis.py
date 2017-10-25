from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
import edward as ed


class FactorAnalysis(object):
    '''Factor analysis models for genotype data'''

    def __init__(self,
                 dataset,
                 k=2,
                 spatial_effect=True,
                 sparse_loadings=True,
                 ):
        '''Intialize factor analysis model

        Args:
            dataset: Dataset
                Dataset object from dataset.py
            k: int
                dimension of loadings
            spatial_effect: bool
                include spatial random effect parameterized as a gaussian process
            sparse_loadings: bool
                assume sparse prior on the loadings
        '''
        self.dataset = dataset
        self.k = k
        self.spatial_effect = spatial_effect
        self.sparse_loadings = sparse_loadings

    def _prior(self):
        '''Defines prior distribution on the loadings

        if the loadings are sparse we follow hierarchal prior from:
        http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1001117
        '''
        with tf.name_scope('prior'):
            if self.sparse_loadings:
                self.tau = tf.nn.softplus(tf.Variable(tf.random_normal([self.dataset.genotypes.n, self.k])))
                self.l = ed.models.Normal(loc=tf.zeros([self.dataset.genotypes.n, self.k]), scale=self.tau)
            else:
                self.l = ed.models.Normal(loc=tf.zeros([self.dataset.genotypes.n, self.k]),
                                          scale=tf.ones([self.dataset.genotypes.n, self.k]))

    def _model(self):
        '''Defines the generative model for the latent variables and data

        spatial model inspired from:
        http://journal.frontiersin.org/article/10.3389/fgene.2012.00254/full

        except we treat the factors as random and integrate them out of the model resulting in
        a linear mixed model with a spatial random effect, a low-rank random effect parametrized by
        the loadings (LL^T) and a noise random effect. We also estimate the spatial covariance kernel whereas
        they fix it. Here are some additional notes clarifying the possible models that can be fit here ...

        - If the factor model has no spatial random effect and the loadings
        are not sparse its PCA.
        - If the factor model has no spatial effect and the loadings are sparse
        then its SFA (sprase factor analysis).
        - If the factor model has a spatial random effect and the loadings are not
        sparse then its a LMM version of the spFA model
        - If the factor model has a spatial random effect and the loadings are sparse
        then its sparse spatial factor analysis (sspFA)
        '''
        with tf.name_scope('likelihood'):
            # std dev of noise variance component
            self.sigma_e = tf.nn.softplus(tf.Variable(tf.random_normal([])))

            # noise covariance component
            v_e = (self.sigma_e * tf.eye(self.dataset.genotypes.n))

            # low rank covariance component
            v_l = tf.matmul(self.l, self.l, transpose_b=True)

            if self.spatial_effect:
                # lengthscale of rbf kernel
                self.alpha = tf.nn.softplus(tf.Variable(tf.random_normal([])))

                # std dev of spatial variance component
                self.sigma_s = tf.nn.softplus(tf.Variable(tf.random_normal([])))

                # placeholder for geographic coordinates
                self.x_ph = tf.placeholder(dtype=tf.float32, shape=(self.dataset.genotypes.n, 2))

                # covariance matrix
                v_s = ed.rbf(self.x_ph, variance=tf.square(self.sigma_s),
                             lengthscale=self.alpha)
                self.v = v_s + v_l + v_e
            else:
                # covariance matrix (no spatial effect)
                self.v = v_l + v_e

            # mvn normal likelihood
            self.y = ed.models.MultivariateNormalTriL(loc=tf.zeros([self.dataset.genotypes.p, self.dataset.genotypes.n]),
                                                      scale_tril=tf.cholesky(self.v))

    def _inference(self):
        '''Define variational distributions / parameters

        TODO:
            * check ql scale is correct way to parameterize
        '''
        with tf.name_scope('variational'):
            if self.sparse_loadings:
                self.ql = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.dataset.genotypes.n, self.k])),
                                           scale=self.tau)
            else:
                self.ql = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.dataset.genotypes.n, self.k])),
                                           scale=tf.nn.softplus(tf.Variable(tf.random_normal([]))))

    def _build_graph(self):
        '''Builds computation graph
        '''
        self._prior()
        self._model()
        self._inference()

    def fit(self, n_iter=250):
        '''Fit model with Variational EM

        TODO:
            * add convergence criteria based on heldout data?
            * add attributes that store loss function
            * add ability to change optimizers or explain default

        Args:
            n_iter: int
                number of epochs of variational em
        '''
        with tf.Session() as sess:

            # build computation graph
            self._build_graph()

            # intialize inference
            if self.spatial_effect:
                inference = ed.KLqp({self.l: self.ql}, {self.y: self.dataset.genotypes.y, self.x_ph: self.dataset.locations.x})
                inference.initialize(n_iter=n_iter)
            else:
                inference = ed.KLqp({self.l: self.ql}, {self.y: self.dataset.genotypes.y})
                inference.initialize(n_iter=n_iter)

            # intialize variables
            sess.run(tf.global_variables_initializer())

            # run inference
            self.loss = np.empty(inference.n_iter, dtype=np.float32)
            for i in range(inference.n_iter):
                info_dict = inference.update()
                inference.print_progress(info_dict)
                self.loss[i] = info_dict['loss']

            # finalize inference
            inference.finalize()

            # extract point estimates
            self.l_hat = self.ql.mean().eval() # posterior mean
            self.sigma_e_hat = self.sigma_e.eval() # mle
            if self.sparse_loadings:
                self.tau_hat = self.tau.eval() # mle
            if self.spatial_effect:
                self.sigma_s_hat = self.sigma_s.eval() # mle
                self.alpha_hat = self.alpha.eval() # mle

