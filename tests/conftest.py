
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import tensorflow as tf
import pytest


class VarmaSim:
    def __init__(self, n=1024):
        self.n = n

    def simData(self, varCoef, vmaCoef, sigma=np.array([1.])):
        n = self.n
        dim = vmaCoef.shape[1]
        lag_ma = vmaCoef.shape[0]
        lag_ar = varCoef.shape[0]

        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma

        x_init = np.array(np.zeros(shape = [lag_ar +1, dim]))
        x = np.empty(( n +101, dim))
        x[:] = np.NaN
        x[:lag_ar +1, ] = x_init
        epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma ,])
        for i in np.arange(lag_ar +1, x.shape[0]):
            epsilon = np.concatenate \
                ([np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[1 ,]), epsilon[:-1]])
            x[i,] = np.sum(np.matmul(varCoef, x[ i -1: i -lag_ar -1:-1][... ,np.newaxis]), axis=(0, -1)) + \
                    np.sum(np.matmul(vmaCoef, epsilon[... ,np.newaxis]), axis=(0, -1))
        x = x[101: ]
        return x

    def calculateSpecMatrix(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse = False):
        specTrue = np.apply_along_axis \
            (lambda f: self.calculateSpecMatrixHelper(f, varCoef, vmaCoef, sigma, inverse=inverse), axis=1, arr = f.reshape(-1 ,1))
        return specTrue

    def calculateSpecMatrixHelper(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse = False):
        # f:     a single frequency
        # Phi:   AR coefficient array. M-by-p-by-p. i-th row imply lag (M-i), p is dim of time series.
        dim = vmaCoef.shape[1]
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma

        k_ar = np.arange(1, varCoef.shape[0 ] +1, 1)
        A_f_re_ar = np.sum(varCoef * np.cos(np.pi * 2 *k_ar*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_im_ar = - np.sum(varCoef * np.sin(np.pi * 2 *k_ar *f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_ar = A_f_re_ar + 1j * A_f_im_ar
        A_bar_f_ar = np.identity(dim) - A_f_ar
        H_f_ar = np.linalg.inv(A_bar_f_ar)

        k_ma = np.arange(vmaCoef.shape[0])
        A_f_re_ma = np.sum(vmaCoef * np.cos(np.pi * 2 *k_ma *f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_im_ma = - np.sum(vmaCoef * np.sin(np.pi * 2 *k_ma *f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_ma = A_f_re_ma + 1j * A_f_im_ma
        A_bar_f_ma = A_f_ma  # + np.identity(dim) already included in the prev A_f_re,im steps
        H_f_ma = A_bar_f_ma

        if inverse == False:
            Spec_mat = H_f_ar @ H_f_ma @ Sigma @ H_f_ma.conj().T @ H_f_ar.conj().T
            return Spec_mat
        else:
            Spec_inv = A_bar_f_ar.conj().T @ A_bar_f_ma.conj().T @ np.linalg.inv(Sigma) @ A_bar_f_ma @ A_bar_f_ar
            return Spec_inv



@pytest.fixture
def Simulation():
    np.random.seed(1234567)

    Simulation = VarmaSim(n=512)

    sigma = np.array([[1., 0.8], [0.8, 1.]])
    var_coef = np.array([[[0.2, 0.5], [0., -0.2]], [[0., 0.], [0.5, -0.2]]])
    vma_coef = np.array([[[1., 0.], [0., 1.]], [[0.6, 0], [0.2, -0.5]],
                         [[0.3, 0], [0, 0.3]]])

    x = Simulation.simData(var_coef, vma_coef, sigma=sigma)
#    freq = np.arange(1, np.floor_divide(500 * 2, 2) + 1, 1) / (500 * 2)
    n = x.shape[0]
    if np.mod(n, 2) == 0:
        # n is even
        freq = np.arange(1, int(n / 2) + 1) / n
    else:
        # n is odd
        freq = np.arange(1, int((n - 1) / 2) + 1) / n
    spec_true = Simulation.calculateSpecMatrix(freq, var_coef, vma_coef, sigma)

    return x, freq, spec_true
