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


class SpecPrep:  # Parent used to create SpecModel object
    def __init__(self, x):
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # ts:     time series x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.ts = x
        if x.shape[1] < 2:
            raise Exception("Time series should be at least 2 dimensional.")

        self.y_ft = []  # inital
        self.freq = []
        self.p_dim = []
        # self.Xmat = []
        self.Zar = []
        # other self variables can be defined later within methods
        # in init, can't use the methods in the class defined below
        # 需要逐层定义，不能跨级，例如self.a.b不可行，必须先定义有b属性的a才可以

    # scaled fft and get the elements of freq = 1:[Nquist]
    # discarding the rest of freqs
    def sc_fft(self):
        # x is a n-by-p matrix
        # unscaled fft
        x = self.ts
        y = np.apply_along_axis(np.fft.fft, 0, x)
        # scale it
        n = x.shape[0]
        y = y / np.sqrt(n)
        # discard 0 freq
        y = y[1:]
        if np.mod(n, 2) == 0:
            # n is even
            y = y[0:int(n / 2)]
            fq_y = np.arange(1, int(n / 2) + 1) / n
        else:
            # n is odd
            y = y[0:int((n - 1) / 2)]
            fq_y = np.arange(1, int((n - 1) / 2) + 1) / n
        p_dim = x.shape[1]

        self.y_ft = y
        self.freq = fq_y
        self.p_dim = p_dim
        self.num_obs = fq_y.shape[0]
        return dict(y=y, fq_y=fq_y, p_dim=p_dim)

    # Demmler-Reinsch basis for linear smoothing splines (Eubank,1999)
    def DR_basis(self, N=10):
        # nu: vector of frequences
        # N:  amount of basis used
        # return a len(nu)-by-N matrix
        nu = self.freq
        basis = np.array([np.sqrt(2) * np.cos(x * np.pi * nu) for x in np.arange(1, N + 1)]).T
        return basis

    #  DR_basis(y_ft$fq_y, N=10)

    # cbinded X matrix
    def Xmtrix(self, N_delta=15, N_theta=15):
        nu = self.freq
        X_delta = np.concatenate([np.column_stack([np.repeat(1, nu.shape[0]), nu]), self.DR_basis(N=N_delta)], axis=1)
        X_theta = np.concatenate([np.column_stack([np.repeat(1, nu.shape[0]), nu]), self.DR_basis(N=N_theta)], axis=1)
        try:
            if self.Xmat_delta is not None:
                Xmat_delta = tf.convert_to_tensor(X_delta, dtype=tf.float32)
                Xmat_theta = tf.convert_to_tensor(X_theta, dtype=tf.float32)
                return Xmat_delta, Xmat_theta
        except:  # NPE
            self.Xmat_delta = tf.convert_to_tensor(X_delta, dtype=tf.float32)  # basis matrix
            self.Xmat_theta = tf.convert_to_tensor(X_theta, dtype=tf.float32)
            self.N_delta = N_delta  # N
            self.N_theta = N_theta
            return self.Xmat_delta, self.Xmat_theta

    # working respose from y
    def y_work(self):
        p_dim = self.p_dim
        y_work = self.y_ft
        for i in np.arange(1, p_dim):
            y_work = np.concatenate([y_work, self.y_ft[:, i:]], axis=1)

        self.y_work = y_work
        return self.y_work

    # use inside Zmtrix
    def dmtrix_k(self, y_k):
        # y_work: N-by-p*(p+1)/2 array, N is #ffreq chosen in sc_fft, p > 1 is dimension of mvts
        # y_k is the k-th splitted row of y_work, y_k is an 1-by- matrix
        p_work = y_k.shape[1]
        p = self.p_dim
        Z_k = np.zeros([p_work, p_work - p], dtype=complex)

        yy_k = y_k[:, np.cumsum(np.concatenate([[0], np.arange(p, 1, -1)]))]
        times = np.arange(p - 1, -1, -1)
        Z_k = block_diag(*[np.diag(np.repeat(yy_k[0, j], times[j]), k=-1)[:, :times[j]] for j in range(p)])
        return Z_k

    # compute Z_ar[k, , ] = Z_k, k = 1,...,#freq in sc_ffit
    def Zmtrix(self):  # dense Z matrix
        # return 3-d array
        y_work = self.y_work()
        n, p = y_work.shape
        if p > 1:
            y_ls = np.split(y_work, n)
            Z_ = np.array([self.dmtrix_k(x) for x in y_ls])
        else:
            Z_ = 0
        self.Zar_re = np.real(Z_)  # add new variables to self, if Zar not defined in init at the beginning
        self.Zar_im = np.imag(Z_)
        return self.Zar_re, self.Zar_im

    # Sparse matrix form of Zmtrix()
    def SparseZmtrix(self):  # sparse Z matrix
        y_work = self.y_work()
        n, p = y_work.shape

        if p == 1:
            raise Exception('To use sparse representation, dimension of time series should be at least 2')
            return

        y_ls = np.split(y_work, n)

        coomat_re_ls = []
        coomat_im_ls = []
        for i in range(n):
            Zar = self.dmtrix_k(y_ls[i])
            Zar_re = np.real(Zar)
            Zar_im = np.imag(Zar)
            coomat_re_ls.append(coo_matrix(Zar_re))
            coomat_im_ls.append(coo_matrix(Zar_im))

        Zar_re_indices = []
        Zar_im_indices = []
        Zar_re_values = []
        Zar_im_values = []
        for i in range(len(coomat_re_ls)):
            Zar_re_indices.append(np.stack([coomat_re_ls[i].row, coomat_re_ls[i].col], -1))
            Zar_im_indices.append(np.stack([coomat_im_ls[i].row, coomat_im_ls[i].col], -1))
            Zar_re_values.append(coomat_re_ls[i].data)
            Zar_im_values.append(coomat_im_ls[i].data)

        self.Zar_re_indices = Zar_re_indices
        self.Zar_im_indices = Zar_im_indices
        self.Zar_re_values = Zar_re_values
        self.Zar_im_values = Zar_im_values
        self.Zar_size = Zar.shape
        return [self.Zar_re_indices, self.Zar_re_values], [self.Zar_im_indices, self.Zar_im_values]

#
