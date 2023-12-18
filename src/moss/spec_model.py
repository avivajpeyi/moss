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
    def __init__(self, x, hyper, sparse_op=False):
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
        ldelta_ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        tmp1_ = - tf.reduce_sum(ldelta_, [1, 2])
        # delta_ = tf.exp(ldelta_)
        delta_inv = tf.exp(- ldelta_)
        theta_re = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0, 2, 1]))  # no need \ here
        theta_im = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        # Z_theta_re = tf.transpose(
        #    tf.linalg.diag_part(tf.transpose( tf.tensordot(self.Z_re, theta_re, [[2],[2]]) - tf.tensordot(self.Z_im, theta_im, [[2],[2]]), perm = (2,1,0,3) ) ), perm = (0,2,1) )
        # Z_theta_im = tf.transpose(
        #    tf.linalg.diag_part(tf.transpose( tf.tensordot(self.Z_re, theta_im, [[2],[2]]) + tf.tensordot(self.Z_im, theta_re, [[2],[2]]), perm = (2,1,0,3) ) ), perm = (0,2,1) )
        Z_theta_re = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), theta_re) - tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), theta_im)
        Z_theta_im = tf.linalg.matvec(tf.expand_dims(self.Z_re, 0), theta_im) + tf.linalg.matvec(
            tf.expand_dims(self.Z_im, 0), theta_re)

        u_re = self.y_re - Z_theta_re
        u_im = self.y_im - Z_theta_im

        tmp2_ = - tf.reduce_sum(tf.multiply(tf.square(u_re) + tf.square(u_im), delta_inv), [1, 2])

        log_lik = tmp1_ + tmp2_
        return log_lik

        # Sparse form of loglik()

    def loglik_sparse(self, params):
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta, xxx,
        #                                       ga_theta_re, xxx,
        #                                       ga_theta_im, xxx, ...)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters

        ldelta_ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        tmp1_ = - tf.reduce_sum(ldelta_, [1, 2])
        # delta_ = tf.exp(ldelta_)
        delta_inv = tf.exp(- ldelta_)
        theta_re = tf.matmul(self.Xmat_theta, tf.transpose(params[2], [0, 2, 1]))  # no need \ here
        theta_im = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        Z_theta_re_ls = [
            tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(theta_re[:, i])) - tf.sparse.sparse_dense_matmul(
                self.Z_im[i], tf.transpose(theta_im[:, i])) for i in range(self.num_obs)]
        Z_theta_im_ls = [
            tf.sparse.sparse_dense_matmul(self.Z_re[i], tf.transpose(theta_im[:, i])) + tf.sparse.sparse_dense_matmul(
                self.Z_im[i], tf.transpose(theta_re[:, i])) for i in range(self.num_obs)]

        u_re = self.y_re - tf.transpose(tf.stack(Z_theta_re_ls), [2, 0, 1])
        u_im = self.y_im - tf.transpose(tf.stack(Z_theta_im_ls), [2, 0, 1])

        tmp2_ = - tf.reduce_sum(tf.multiply(tf.square(u_re) + tf.square(u_im), delta_inv), [1, 2])
        log_lik = tmp1_ + tmp2_
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


class SpecVI:
    def __init__(self, x):
        self.data = x

    def runModel(self, N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, inference_size=500,
                 inference_freq=(np.arange(1, 500 + 1, 1) / (500 * 2)),
                 variation_factor=0, sparse_op=False):
        self.sparse_op = sparse_op

        x = self.data
        print('data shape: ' + str(x.shape))

        ## Hyperparameter
        ##
        hyper_hs = []
        tau0 = 0.01
        c2 = 4
        sig2_alp = 10
        degree_fluctuate = N_delta / 2  # the smaller tends to be smoother
        hyper_hs.extend([tau0, c2, sig2_alp, degree_fluctuate])

        ## Define Model
        ##
        Spec_hs = SpecModel(x, hyper_hs, sparse_op=self.sparse_op)
        self.model = Spec_hs  # save model object
        # comput fft
        Spec_hs.sc_fft()
        # compute array of design matrix Z, 3d
        if self.sparse_op == False:
            Spec_hs.Zmtrix()
        else:
            Spec_hs.SparseZmtrix()
        # compute X matrix related to basis function on ffreq
        Spec_hs.Xmtrix(N_delta, N_theta)
        # convert all above to tensorflow object
        Spec_hs.toTensor()
        # create tranable variables
        Spec_hs.createModelVariables_hs()

        print('Start Model Inference Training: ')

        '''
        # Phase1 obtain MAP
        '''
        lr = lr_map
        n_train = ntrain_map  #
        optimizer_hs = tf.keras.optimizers.Adam(lr)

        start_total = timeit.default_timer()
        start_map = timeit.default_timer()

        # train
        @tf.function
        def train_hs(model, optimizer, n_train):
            # model:    model object
            # optimizer
            # n_train:  times of training
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

            for i in tf.range(n_train):
                if self.sparse_op == False:
                    lpost = model.train_one_step(optimizer, model.loglik, model.logprior_hs)
                else:
                    lpost = model.train_one_step(optimizer, model.loglik_sparse, model.logprior_hs)
                if optimizer.iterations % 500 == 0:
                    tf.print('Step', optimizer.iterations, ': log posterior', lpost)
                lp = lp.write(tf.cast(i, tf.int32), lpost)
            return model.trainable_vars, lp.stack()

        print('Start Point Estimating: ')
        opt_vars_hs, lp_hs = train_hs(Spec_hs, optimizer_hs, n_train)
        # opt_vars_hs:         self.trainable_vars(ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau)
        # Variational inference for regression parameters
        end_map = timeit.default_timer()
        print('MAP Training Time: ', end_map - start_map)
        self.lp = lp_hs

        idx = tf.where(tf.reduce_sum(tf.cast(self.model.trainable_vars[2][0, :, 2:] >= 0.1, tf.int32) + tf.cast(
            self.model.trainable_vars[4][0, :, 2:] >= 0.1, tf.int32), -1) == 0)
        for i in idx:
            self.model.trainable_vars[2][0, i[0]].assign(tf.zeros(self.model.trainable_vars[2][0, i[0]].shape))
            self.model.trainable_vars[4][0, i[0]].assign(tf.zeros(self.model.trainable_vars[4][0, i[0]].shape))
        opt_vars_hs = self.model.trainable_vars
        '''
        Phase 2 UQ
        '''
        optimizer_vi = tf.optimizers.Adam(5e-2)
        if variation_factor <= 0:
            trainable_Mvnormal = tfd.JointDistributionSequential([
                tfd.Independent(
                    tfd.MultivariateNormalDiag(loc=opt_vars_hs[i][0],
                                               scale_diag=tfp.util.TransformedVariable(
                                                   tf.constant(1e-4, tf.float32, opt_vars_hs[i][0].shape),
                                                   tfb.Softplus(), name='q_z_scale')),
                    reinterpreted_batch_ndims=1)
                for i in tf.range(len(opt_vars_hs))])
        else:  # variation_factor > 0
            trainable_Mvnormal = tfd.JointDistributionSequential([
                tfd.Independent(
                    tfd.MultivariateNormalDiagPlusLowRank(loc=opt_vars_hs[i][0],
                                                          scale_diag=tfp.util.TransformedVariable(
                                                              tf.constant(1e-4, tf.float32, opt_vars_hs[i][0].shape),
                                                              tfb.Softplus()),
                                                          scale_perturb_factor=tfp.util.TransformedVariable(
                                                              tf.random_uniform_initializer()(
                                                                  opt_vars_hs[i][0].shape + variation_factor),
                                                              tfb.Identity())),
                    reinterpreted_batch_ndims=1)
                for i in tf.range(len(opt_vars_hs))])

        if self.sparse_op == False:
            def conditioned_log_prob(*z):
                return Spec_hs.loglik(z) + Spec_hs.logprior_hs(z)
        else:
            def conditioned_log_prob(*z):
                return Spec_hs.loglik_sparse(z) + Spec_hs.logprior_hs(z)

        print('Start UQ training: ')
        start = timeit.default_timer()
        losses = tf.function(
            lambda l: tfp.vi.fit_surrogate_posterior(target_log_prob_fn=l, surrogate_posterior=trainable_Mvnormal,
                                                     optimizer=optimizer_vi, num_steps=500 * 2))(
            conditioned_log_prob)  #
        stop = timeit.default_timer()
        print('VI Time: ', stop - start)
        stop_total = timeit.default_timer()
        self.kld = losses
        # plt.plot(losses)
        ##
        ## Phase 3 (Optional) In our case, can be skipped since effects very little
        ##
        # =============================================================================
        # scale_init = [e.read_value() for e in trainable_Mvnormal.trainable_variables]
        # optimizer_vi = tf.optimizers.Adam(5e-3)
        # trainable_Mvnormal_p3 = tfd.JointDistributionSequential([
        #     tfd.Independent(
        #     tfd.MultivariateNormalDiag(loc = tf.Variable(opt_vars_hs[i][0], name='q_z_loc'),
        #                                scale_diag = tfp.util.TransformedVariable(tfb.Softplus()(scale_init[i]),
        #                                                                          tfb.Softplus(), name='q_z_scale')) ,
        #     reinterpreted_batch_ndims=1)
        #     for i in tf.range(len(opt_vars_hs))])
        #
        # def conditioned_log_prob(*z):
        #     return Spec_hs.loglik(z) + Spec_hs.logprior_hs(z)
        #
        # print('Start Phase 3 fine-tuning: ')
        # start = timeit.default_timer()
        # losses = tf.function(lambda l: tfp.vi.fit_surrogate_posterior(target_log_prob_fn=l, surrogate_posterior=trainable_Mvnormal, optimizer=optimizer_vi, num_steps=500))(conditioned_log_prob) #
        # stop = timeit.default_timer()
        # print('Fine-tuning Time: ', stop - start)
        # stop_total = timeit.default_timer()
        # plt.plot(losses)
        # =============================================================================
        print('Total Inference Training Time: ', stop_total - start_total)

        self.posteriorPointEst = trainable_Mvnormal.mean()
        self.posteriorPointEstStd = trainable_Mvnormal.stddev()
        self.variationalDistribution = trainable_Mvnormal

        samp = trainable_Mvnormal.sample(inference_size)
        Spec_hs.freq = inference_freq
        Xmat_delta, Xmat_theta = Spec_hs.Xmtrix(N_delta=N_delta, N_theta=N_theta)
        Spec_hs.toTensor()

        delta2_all_s = tf.exp(tf.matmul(Xmat_delta, tf.transpose(samp[0], [0, 2, 1])))
        theta_re_s = tf.matmul(Xmat_theta, tf.transpose(samp[2], [0, 2, 1]))
        theta_im_s = tf.matmul(Xmat_theta, tf.transpose(samp[4], [0, 2, 1]))
        theta2_s = tf.square(theta_re_s) + tf.square(theta_im_s)

        delta_split_index = np.cumsum(np.arange(Spec_hs.p_dim, 1, -1))
        theta_split_index = np.cumsum(np.arange(Spec_hs.p_dim - 1, 1, -1))

        delta2_ls = np.split(delta2_all_s, delta_split_index, axis=2)
        theta_re_ls = np.split(theta_re_s, theta_split_index, axis=2)
        theta_im_ls = np.split(theta_im_s, theta_split_index, axis=2)
        theta2_ls = np.split(theta2_s, theta_split_index, axis=2)

        Spec_density_quant = []
        Offdiag_re_quant = []
        Offdiag_im_quant = []
        Sq_Coherence_quant = []
        for i in range(len(delta2_ls)):
            if i == Spec_hs.p_dim - 1:
                delta2_all_s = delta2_ls[i]
                delta2_s = delta2_all_s[..., 0:1]
                Spec_density_s = delta2_s
                Spec_density_q = tfp.stats.percentile(Spec_density_s, [2.5, 50, 97.5], axis=0)
                Spec_density_quant.append(Spec_density_q)
            else:
                delta2_all_s = delta2_ls[i]
                delta2_s = delta2_all_s[..., 0:1]
                theta_re_s = theta_re_ls[i]
                theta_im_s = theta_im_ls[i]
                theta2_s = theta2_ls[i]
                delta2_theta2_s = tf.multiply(delta2_s, theta2_s)

                Spec_density_s = tf.concat([delta2_s, delta2_theta2_s + delta2_all_s[..., 1:]], axis=-1)
                Spec_density_q = tfp.stats.percentile(Spec_density_s, [2.5, 50, 97.5], axis=0)
                Offdiag_re_s = tf.multiply(delta2_s, theta_re_s)
                Offdiag_re_q = tfp.stats.percentile(Offdiag_re_s, [2.5, 50, 97.5], axis=0)
                Offdiag_im_s = tf.multiply(delta2_s, theta_im_s)
                Offdiag_im_q = tfp.stats.percentile(Offdiag_im_s, [2.5, 50, 97.5], axis=0)
                Sq_Coherence_s = tf.divide(delta2_theta2_s, delta2_theta2_s + delta2_all_s[..., 1:])
                Sq_Coherence_q = tfp.stats.percentile(Sq_Coherence_s, [2.5, 50, 97.5], axis=0)

                Spec_density_quant.append(Spec_density_q)
                Offdiag_re_quant.append(Offdiag_re_q)
                Offdiag_im_quant.append(Offdiag_im_q)
                Sq_Coherence_quant.append(Sq_Coherence_q)

        '''
        Obtain [500, p, p] spectral matrices. Here we fix the resolution to be 500 (high enough)
        For simplicity, the off-diag entries are absolute value of original complex entries.
        '''
        self.spec_matrix_est = self.get_SpecMatEst(Spec_density_quant, Sq_Coherence_quant)

        return self.spec_matrix_est, Spec_hs, Spec_density_quant, Sq_Coherence_quant, Offdiag_re_quant, Offdiag_im_quant

    def get_SpecMatEst(self, Spec_density_ls, Sq_Coherence_ls):
        try:
            b = [x.numpy() for x in Spec_density_ls]
            a = [x.numpy() for x in Sq_Coherence_ls]
        except:
            b = Spec_density_ls
            a = Sq_Coherence_ls
        p_ts = len(b)
        Spec_mat = np.zeros([b[0].shape[0], b[0].shape[1], p_ts, p_ts])
        for q in range(b[0].shape[0]):
            for i in range(p_ts):
                Spec_mat[q, ..., i, i] = b[i][q, ..., 0]
            for i in range(0, p_ts - 1):
                for j in range(i + 1, p_ts):
                    Spec_mat[q, ..., i, j] = np.sqrt(
                        a[i][q][..., j - i - 1] * Spec_mat[q, ..., i, i] * Spec_mat[q, ..., j, j])
                    Spec_mat[q, ..., j, i] = Spec_mat[q, ..., i, j]

        # guarantee to be positive-definite matrices
        for q in range(Spec_mat.shape[0]):
            for n in range(Spec_mat.shape[1]):
                d, v = np.linalg.eig(Spec_mat[q, n])
                d = np.maximum(d, 1e-8)
                values = v @ np.apply_along_axis(np.diag, axis=-1, arr=d) @ np.linalg.inv(v)
                Spec_mat[q, n] = values

        return Spec_mat
