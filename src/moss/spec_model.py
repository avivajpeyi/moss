# -*- coding: utf-8 -*-
"""
Define spectral model class

@author: Zhixiong Hu, UCSC
"""
import timeit
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from .spec_prep import SpecPrep


#
## Spectrum Model, subclass of SpecPrep 继承了inital以及所有的methods
#
class SpecModel(SpecPrep):
    def __init__(self, x, hyper, sparse_op=False, **kwargs):
        super().__init__(x)
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # hyper:  list of hyperparameters for prior
        # ts:     time series == x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.hyper = hyper
        self.sparse_op = sparse_op
        self.trainable_vars = []  # all trainable variables

    def __repr__(self):
        return f"SpecModel(ndata={len(self.ts)}, p_dim={self.p_dim})"

    def toTensor(self):
        # convert to tensorflow object
        self.ts = tf.convert_to_tensor(self.ts, dtype=tf.float32)
        self.y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_work = tf.convert_to_tensor(self.y_work, dtype=tf.complex64)
        self.y_re = tf.math.real(self.y_work)  # not y_ft
        self.y_im = tf.math.imag(self.y_work)
        self.freq = tf.convert_to_tensor(self.freq, dtype=tf.float32)
        self.p_dim = tf.convert_to_tensor(self.p_dim, dtype=tf.int32)
        self.N_delta = tf.convert_to_tensor(self.N_delta, dtype=tf.int32)
        self.N_theta = tf.convert_to_tensor(self.N_theta, dtype=tf.int32)
        self.Xmat_delta = tf.convert_to_tensor(self.Xmat_delta, dtype=tf.float32)
        self.Xmat_theta = tf.convert_to_tensor(self.Xmat_theta, dtype=tf.float32)

        if self.sparse_op == False:
            self.Zar = tf.convert_to_tensor(self.Zar, dtype=tf.complex64)  # complex array
            self.Z_re = tf.convert_to_tensor(self.Zar_re, dtype=tf.float32)
            self.Z_im = tf.convert_to_tensor(self.Zar_im, dtype=tf.float32)
        else:  # sparse_op == True
            self.Zar_re_indices = [tf.convert_to_tensor(x, tf.int64) for x in
                                   self.Zar_re_indices]  # int64 required by tf.sparse.SparseTensor
            self.Zar_im_indices = [tf.convert_to_tensor(x, tf.int64) for x in self.Zar_im_indices]
            self.Zar_re_values = [tf.convert_to_tensor(x, tf.float32) for x in self.Zar_re_values]
            self.Zar_im_values = [tf.convert_to_tensor(x, tf.float32) for x in self.Zar_im_values]

            self.Zar_size = tf.convert_to_tensor(self.Zar_size, tf.int64)

            self.Z_re = [tf.sparse.SparseTensor(x, y, self.Zar_size) for x, y in
                         zip(self.Zar_re_indices, self.Zar_re_values)]
            self.Z_im = [tf.sparse.SparseTensor(x, y, self.Zar_size) for x, y in
                         zip(self.Zar_im_indices, self.Zar_im_values)]

        self.hyper = [tf.convert_to_tensor(self.hyper[i], dtype=tf.float32) for i in range(len(self.hyper))]
        if self.p_dim > 1:
            self.n_theta = tf.cast(self.p_dim * (self.p_dim - 1) / 2, tf.int32)  # number of theta in the model

    def createModelVariables_hs(self, batch_size=1):
        #
        #
        # rule:  self.trainable_vars[0, 2, 4] must be corresponding spline regression parameters for p_dim>1
        # in 1-d case, self.trainable_vars[0] must be ga_delta parameters, no ga_theta included.

        # initial values are quite important for training
        p = int(self.y_work.shape[1])
        size_delta = int(self.Xmat_delta.shape[1])
        size_theta = int(self.Xmat_theta.shape[1])

        # initializer = tf.initializers.GlorotUniform() # xavier initializer
        # initializer = tf.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        # initializer = tf.initializers.zeros()

        # better to have deterministic inital on reg coef to control
        ga_initializer = tf.initializers.zeros()
        if size_delta <= 10:
            cvec_d = 0.
        else:
            cvec_d = tf.concat([tf.zeros(10 - 2) + 0., tf.zeros(size_delta - 10) + 1.], 0)
        if size_theta <= 10:
            cvec_o = 0.5
        else:
            cvec_o = tf.concat([tf.zeros(10) + 0.5, tf.zeros(size_theta - 10) + 1.5], 0)

        ga_delta = tf.Variable(ga_initializer(shape=(batch_size, p, size_delta), dtype=tf.float32), name='ga_delta')
        lla_delta = tf.Variable(ga_initializer(shape=(batch_size, p, size_theta - 2), dtype=tf.float32) - cvec_d,
                                name='lla_delta')
        ltau = tf.Variable(ga_initializer(shape=(batch_size, p, 1), dtype=tf.float32) - 1, name='ltau')
        self.trainable_vars.append(ga_delta)
        self.trainable_vars.append(lla_delta)

        nn = int(self.n_theta)  # number of thetas in the model
        ga_theta_re = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype=tf.float32),
                                  name='ga_theta_re')
        ga_theta_im = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype=tf.float32),
                                  name='ga_theta_im')

        lla_theta_re = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype=tf.float32) - cvec_o,
                                   name='lla_theta_re')
        lla_theta_im = tf.Variable(ga_initializer(shape=(batch_size, nn, size_theta), dtype=tf.float32) - cvec_o,
                                   name='lla_theta_im')

        ltau_theta = tf.Variable(ga_initializer(shape=(batch_size, nn, 1), dtype=tf.float32) - 1.5, name='ltau_theta')

        self.trainable_vars.append(ga_theta_re)
        self.trainable_vars.append(lla_theta_re)
        self.trainable_vars.append(ga_theta_im)
        self.trainable_vars.append(lla_theta_im)

        self.trainable_vars.append(ltau)
        self.trainable_vars.append(ltau_theta)

        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)

    def loglik(self, params):  # log-likelihood for mvts p_dim > 1
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta, xxx,
        #                                       ga_theta_re, xxx,
        #                                       ga_theta_im, xxx, ...)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters

        assert self.Xmat_delta.shape[-1] == params[0].shape[-1]
        xγ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        sum_xγ = - tf.reduce_sum(xγ, [1, 2])
        exp_xγ_inv = tf.exp(- xγ)

        xα = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0, 2, 1]))  # no need \ here
        xβ = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        # Z = Sum [(xα + i xβ) * y]
        Z_theta_re = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), xα) - tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), xβ)
        Z_theta_im = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), xβ) + tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), xα)

        u_re = self.y_re - Z_theta_re
        u_im = self.y_im - Z_theta_im

        numerator = tf.square(u_re) + tf.square(u_im)
        internal = tf.multiply(numerator, exp_xγ_inv)
        tmp2_ = - tf.reduce_sum(internal, [-2, -1]) # sum over p_dim and freq
        log_lik = tf.reduce_sum(sum_xγ + tmp2_) # sum over all LnL
        return log_lik

        # Sparse form of loglik()

    def loglik_sparse(self, params):
        """

        y_re:            self.y_re
        y_im:            self.y_im
        Z_:              self.Zar
        X_:              self.Xmat
        params:          self.trainable_vars (ga_delta, xxx,
                                              ga_theta_re, xxx,
                                              ga_theta_im, xxx, ...)
        each of params is a 3-d tensor with sample_size as the fist dim.
        self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters


            .. math::

            \log like(y|params)
                = \sum_{k=1}^{Nyquist Freq}   xγ + abs(  + y - sum( Z * y) )**2 / exp(xγ)

        :param params:
        :return:
        """


        xγ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        sum_xγ = - tf.reduce_sum(xγ, [1, 2])
        exp_xγ_inv = tf.exp(- xγ)

        xα = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0, 2, 1]))  # no need \ here
        xβ = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        # Sum (xα * Z_re - xβ * Z_im) * y_re
        sum_xα_xβ_dot_y_real = [
            tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(xα[:, i])) - tf.sparse.sparse_dense_matmul(
                self.Z_im[i], tf.transpose(xβ[:, i])) for i in range(self.num_obs)]
        sum_xα_xβ_dot_y_imag = [
            tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(xβ[:, i])) + tf.sparse.sparse_dense_matmul(
                self.Z_im[i], tf.transpose(xα[:, i])) for i in range(self.num_obs)]

        u_re = self.y_re - tf.transpose(tf.stack(sum_xα_xβ_dot_y_real), [2, 0, 1])
        u_im = self.y_im - tf.transpose(tf.stack(sum_xα_xβ_dot_y_imag), [2, 0, 1])

        internal = tf.multiply(tf.square(u_re) + tf.square(u_im), exp_xγ_inv),
        tmp2_ = - tf.reduce_sum(internal, [1, 2])
        log_lik = sum_xγ + tmp2_
        return log_lik

    #
    # Model training one step
    #
    def train_one_step(self, optimizer, loglik, prior):  # one step training
        with tf.GradientTape() as tape:
            loss = - loglik(self.trainable_vars) - prior(self.trainable_vars)  # negative log posterior
        grads = tape.gradient(loss, self.trainable_vars)
        optimizer.apply_gradients(zip(grads, self.trainable_vars))
        return - loss  # return log posterior

    # For new prior strategy, need new createModelVariables() and logprior()

    def logprior_hs(self, params):
        # hyper:           list of hyperparameters (tau0, c2, sig2_alp, degree_fluctuate)
        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters
        Sigma1 = tf.multiply(tf.eye(tf.constant(2), dtype=tf.float32), self.hyper[2])
        priorDist1 = tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(Sigma1))  # can also use tfd.MultivariateNormalDiag

        Sigm = tfb.Sigmoid()
        s_la_alp = Sigm(- tf.range(1, params[1].shape[-1] + 1., dtype=tf.float32) + self.hyper[3])
        priorDist_la_alp = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_alp)

        s_la_theta = Sigm(- tf.range(1, params[3].shape[-1] + 1., dtype=tf.float32) + self.hyper[3])
        priorDist_la_theta = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_theta)

        a2 = tf.square(tf.exp(params[1]))
        Sigma2i_diag = tf.divide(tf.multiply(tf.multiply(a2, tf.square(tf.exp(params[6]))), self.hyper[1]),
                                 tf.multiply(a2, tf.square(tf.exp(params[6]))) + self.hyper[1])

        priorDist2 = tfd.MultivariateNormalDiag(scale_diag=Sigma2i_diag)

        lpriorAlp_delt = tf.reduce_sum(priorDist1.log_prob(params[0][:, :, 0:2]), [1])  #
        lprior_delt = tf.reduce_sum(priorDist2.log_prob(params[0][:, :, 2:]),
                                    [1])  # only 2 dim due to log_prob rm the event_shape dim
        lpriorla_delt = tf.reduce_sum(priorDist_la_alp.log_prob(tf.exp(params[1])), [1, 2]) + tf.reduce_sum(params[1],
                                                                                                            [1, 2])
        lpriorDel = lprior_delt + lpriorla_delt + lpriorAlp_delt

        a3 = tf.square(tf.exp(params[3]))
        Sigma3i_diag = tf.divide(tf.multiply(tf.multiply(a3, tf.square(tf.exp(params[7]))), self.hyper[1]),
                                 tf.multiply(a3, tf.square(tf.exp(params[7]))) + self.hyper[1])

        priorDist3 = tfd.MultivariateNormalDiag(scale_diag=Sigma3i_diag)

        lprior_thet_re = tf.reduce_sum(priorDist3.log_prob(params[2]), [1])
        lpriorla_thet_re = tf.reduce_sum(priorDist_la_theta.log_prob(tf.exp(params[3])), [1, 2]) + tf.reduce_sum(
            params[3], [1, 2])
        lpriorThe_re = lprior_thet_re + lpriorla_thet_re

        a4 = tf.square(tf.exp(params[5]))
        Sigma4i_diag = tf.divide(tf.multiply(tf.multiply(a4, tf.square(tf.exp(params[7]))), self.hyper[1]),
                                 tf.multiply(a4, tf.square(tf.exp(params[7]))) + self.hyper[1])

        priorDist4 = tfd.MultivariateNormalDiag(scale_diag=Sigma4i_diag)

        lprior_thet_im = tf.reduce_sum(priorDist4.log_prob(params[4]), [1])
        lpriorla_thet_im = tf.reduce_sum(priorDist_la_theta.log_prob(tf.exp(params[5])), [1, 2]) + tf.reduce_sum(
            params[5], [1, 2])
        lpriorThe_im = lprior_thet_im + lpriorla_thet_im

        priorDist_tau = tfd.HalfCauchy(tf.constant(0, tf.float32), self.hyper[0])
        logPrior = lpriorDel + lpriorThe_re + lpriorThe_im + tf.reduce_sum(
            priorDist_tau.log_prob(tf.exp(params[6])) + params[6], [1, 2]) + tf.reduce_sum(
            priorDist_tau.log_prob(tf.exp(params[7])) + params[7], [1, 2])
        return logPrior

