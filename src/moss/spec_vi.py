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

from .spec_model import SpecModel

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors





class SpecVI:
    def __init__(self, x, model_generator=SpecModel):
        self.data = x
        self.SpecModelGenerator = model_generator

    def runModel(self, N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, inference_size=500,
                 inference_freq=(np.arange(1, 500 + 1, 1) / (500 * 2)),
                 variation_factor=0, sparse_op=False, nchunks=None):
        """

        :param N_delta:
        :param N_theta:
        :param lr_map:
        :param ntrain_map:
        :param inference_size:
        :param inference_freq:
        :param variation_factor:
        :param sparse_op:
        :return:
        """
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
        Spec_hs = self.SpecModelGenerator(x, hyper_hs, sparse_op=self.sparse_op, nchunks=nchunks)
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
        print(f'USING N-CHUNKS: {nchunks} ')

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
