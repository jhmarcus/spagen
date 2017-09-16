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
                 data,
                 k=2,
                 spatial_effect=True,
                 sparse_loadings=True,
                 train_proportion=.8
                 ):
        '''Intialize factor analysis model

        * If the factor model has no spatial random effect and the loadings
        are not sparse its PCA.
        * If the factor model has no spatial effect and the loadings are sparse
        then its SFA (sprase factor analysis).
        * If the factor model has a spatial random effect and the loadings are not
        sparse then its a LMM version of the spFA model
        * If the factor model has a spatial random effect and the loadings are sparse
        then its sparse spatial factor analysis (SSFA)

        Args:
            data: Data
                Data object from data.py
            k: int
                dimension of l_i
            spatial_effect: bool
                include spatial random effect parameterized as a gaussian process
            sparse_loadings: bool
                assume sparse prior on the loadings
            train_proportion: float
                proportion of SNPs to sample into training dataset
        '''
        self.data = data
        self.k = k
        self.spatial_effect = spatial_effect
        self.sparse_loadings = sparse_loadings
        self.p_train = int(train_proportion * self.data.p)

    def _split_data(self):
        '''Randomly splits data into train and validation set
        '''
        # sample indicies
        i = np.random.randint(0, self.data.p, self.p_train)

        # train set
        self.y_train = self.y[i,:]

        # validation set
        mask = np.ones(self.data.p, dtype=bool)
        mask[i] = False
        self.y_val = self.y[mask,:]
        self.p_val = self.y_val.shape[0]

    def _prior(self):
        '''Defines prior distribution on the loadings

        if the loadings are sparse we follow hierarchal prior from
        http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1001117
        '''
        if self.sparse_loadings:
            self.sigma_l = tf.nn.softplus(tf.Variable(tf.random_normal([self.data.n, self.k])))
            self.l = ed.models.Normal(loc=tf.zeros([self.data.n, self.k]), scale=self.sigma_l)
        else:
            self.l = ed.models.Normal(loc=tf.zeros([self.data.n, self.k]), scale=tf.ones([self.data.n, self.k]))

    def _model(self):
        '''Defines the generative model for the latent variables and data
        '''
        # std dev of noise variance component
        self.sigma_e = tf.nn.softplus(tf.Variable(tf.random_normal([])))

        # noise covariance component
        v_e = (self.sigma_e * tf.eye(self.n))

        # low rank covariance component
        v_l = tf.matmul(self.l, self.l, transpose_b=True)

        if self.spatial_effect:
            # lengthscale of rbf kernel
            self.alpha = tf.nn.softplus(tf.Variable(tf.random_normal([])))

            # std dev of spatial variance component
            self.sigma_s = tf.nn.softplus(tf.Variable(tf.random_normal([])))

            # placeholder for geographic coordinates
            self.x_ph = tf.placeholder(dtype=tf.float32, shape=(self.data.n, 2))

            # covariance matrix
            v_s = ed.rbf(self.x_ph, variance=tf.square(self.sigma_s), lengthscale=self.alpha)
            self.v = v_s + v_l + v_e
        else:
            # covariance matrix (no spatial effect)
            self.v = v_l + v_e

        # mvn normal likelihood
        self.y = ed.models.MultivariateNormalTriL(loc=tf.zeros([self.p_train, self.data.n]),
                                                  scale_tril=tf.cholesky(self.v))

    def _validation_log_likelihood(self):
        '''Computes log likelihood of validation set given current parameter estimates
        '''
        # noise covariance estimate
        v_e = self.sigma_e.eval() * np.eye(self.data.n)

        # low-rank covariance estimate
        l_hat = self.l.mean().eval()
        v_l_hat = np.dot(l_hat, l_hat.T)

        if self.spatial_effect:
            # spatial covariance estimate
            v_s_hat = ed.rbf(self.data.x, variance=tf.square(self.sigma_s.eval()),
                             lengthscale=self.alpha.eval())
            v_hat = v_s_hat + v_l_hat + v_e_hat
        else:
            v_hat = v_l_hat + v_hat

        # compute log-likeihood of each SNP in validation set
        lls = np.apply_along_axis(func1d=stats.multivariate_normal.logpdf,
                                  axis=1, arr=self.data.y_val, 0.0, v_hat)

        # compute log-likeihood of all the snps
        ll = lls.sum()

        return(ll)

    def _inference(self, n_iter=10, learning_rate=1e-4, tol=1e-4):
        '''Run black-box Variational EM algorithim

        The posterior distribution of the loadings are estimated via black-box variational
        inference wheras the hyperparameters are estimated through maximum likelihood

        TODO:
            * add convergence criteria based on heldout data?
            * check ql scale is correct way to parameterize
            * add attributes that store loss function
            * add ability to change optimizers or explain default

        Args:
            n_iter: int
                number of epochs of variational em
            learning_rate: float
                learning rate for stochastic graident descent of ELBO
            tol: float
                tolerence for convergence when change in validation
                log-likelihood is small
        '''
        self.ql = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.data.n, self.k])),
                                   scale=self.sigma_l)

        if self.spatial_effect:
            inference = ed.KLqp({self.l: self.ql}, {self.y: self.y_train, self.x_ph: self.data.x})
            inference.initialize(n_print=10, n_iter=n_iter)
        else:
            inference = ed.KLqp({self.l: self.ql}, {self.y: self.data.y})
            inference.initialize(n_print=10, n_iter=n_iter)

        tf.global_variables_initializer().run()

        for i in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)

    def fit(self, n_iter=500, learning_rate=1e-4, tol=1e-4):
        '''Public method to peform inference uder defined priors and model

        Args:
            n_iter: int
                number of epochs of variational em
            learning_rate: float
                learning rate for stochastic VI
            tol: float
                tolerence of convergences when change in validation
                log-likelihood is small
        '''
        self._prior()
        self._model()
        self._inference(n_iter, learning_rate)

