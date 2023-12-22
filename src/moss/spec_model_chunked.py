from .spec_model import SpecModel

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix


import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class SpecModelChunked(SpecModel):
    """Chunked version of SpecModel"""

    # scaled fft and get the elements of freq = 1:[Nquist]
    # discarding the rest of freqs

    def __init__(self, x, hyper_hs, sparse_op=False, nchunks=1):
        super().__init__(x, hyper_hs, sparse_op=sparse_op)
        self.nchunks = nchunks

    def __repr__(self):
        return f"SpecModelChunked(nchunks={self.nchunks}, ndata={len(self.ts)}, p_dim={self.p_dim})"

    #---------------------------------------------------------
    # SPEC_PREP_CHUNKED
    def sc_fft(self):
        # x is a n-by-p matrix
        # unscaled fft
        x = self.ts
        num_segments = self.nchunks
        x = np.array(np.split(x, num_segments))

        y = []
        for i in range(num_segments):
            y_fft = np.apply_along_axis(np.fft.fft, 0, x[i])
            y.append(y_fft)
        y = np.array(y)

        # scale it
        n = x.shape[1]
        y = y / np.sqrt(n)  # np.sqrt(n)
        # discard 0 freq

        Ts = 1
        fq_y = np.fft.fftfreq(np.size(x, 1), Ts)

        if np.mod(n, 2) == 0:
            # n is even
            y = y[:, 1:int(n / 2)+1, :]
            fq_y = fq_y[1:int(n / 2)+1]
        else:
            # n is odd
            y = y[:, 1:int((n - 1) / 2)+1, :]
            fq_y = fq_y[1:int((n - 1) / 2)+1]
        p_dim = x.shape[2]

        self.y_ft = y
        self.freq = fq_y
        self.p_dim = p_dim
        self.num_obs = fq_y.shape[0]
        return dict(y=y, fq_y=fq_y, p_dim=p_dim)

    # working respose from y
    def y_work(self):
        p_dim = self.p_dim
        y_work = self.y_ft
        for i in np.arange(1, p_dim):
            y_work = np.concatenate([y_work, self.y_ft[:, :, i:]], axis=2)

        self.y_work = y_work
        return y_work

    def dmtrix_k(self, y_k):
        # y_work: N-by-p*(p+1)/2 array, N is #ffreq chosen in sc_fft, p > 1 is dimension of mvts
        # y_k is the k-th splitted row of y_work, y_k is an 1-by- matrix
        n, p_work = y_k.shape
        p = self.p_dim
        Z_k = np.zeros([n, p_work, p_work - p], dtype=complex)

        yy_k = y_k[:, np.cumsum(np.concatenate([[0], np.arange(p, 1, -1)]))]
        times = np.arange(p - 1, -1, -1)

        for i in range(n):
            Z_k[i] = block_diag(*[np.diag(np.repeat(yy_k[i, j], times[j]), k=-1)[:, :times[j]] for j in range(p)])
        return Z_k

    # compute Z_ar[k, , ] = Z_k, k = 1,...,#freq in sc_ffit
    def Zmtrix(self):  # dense Z matrix
        # return 3-d array
        y_work = self.y_work()
        c, n, p = y_work.shape
        if p > 1:
            y_ls = np.squeeze(np.split(y_work, c))
            if c==1:
                y_ls = np.expand_dims(y_ls, axis=0)
            Z_ = np.array([self.dmtrix_k(x) for x in y_ls])
        else:
            Z_ = 0
        self.Zar_re = np.real(Z_)  # add new variables to self, if Zar not defined in init at the beginning
        self.Zar_im = np.imag(Z_)

        exected_shape = (c, n, p, (p - 1)/2)
        assert self.Zar_re.shape == exected_shape, f"Zar_re.shape: {self.Zar_re.shape} != {exected_shape}"
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


    def createModelVariables_hs(self, batch_size=1, seed=None):
        #
        #
        # rule:  self.trainable_vars[0, 2, 4] must be corresponding spline regression parameters for p_dim>1
        # in 1-d case, self.trainable_vars[0] must be ga_delta parameters, no ga_theta included.

        # initial values are quite important for training
        p = int(self.y_work.shape[2])
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

