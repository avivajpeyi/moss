# -*- coding: utf-8 -*-
"""
Simulate VAR(3) time series. 
VAR(3) coefficients are from var_coefs.pkl file.

@author: Zhixiong Hu, UCSC
"""

import pickle
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import tensorflow as tf

class VarSim:
    def __init__(self, var_coefs, n=1024, sigma=1.):
        self.n = n
        self.sigma = np.array(sigma).reshape(-1)
        self.coefs = var_coefs
        

    def getData(self):
        x = []
        for coefs in self.coefs:
            xTemp = self.simulateVar(n=self.n, Phi = coefs, sigma=self.sigma)
            x.append(xTemp)
             
        x_all = np.concatenate(x, axis=1)
        return x_all

    # Mvar with white noise
    def simulateVar(self, Phi, n=1024, sigma = np.array([1.])):
        # Phi:   AR coefficient array. M0-by-p-by-p. i-th row imply lag (M-i), p is dim of time series.
        # n:     time stamps  
        # sigma: white noise variance
        dim = Phi.shape[1]
        lag = Phi.shape[0]
        
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        x_init = np.array(np.zeros(shape = [lag+1, dim]))
        x = np.empty((n+101, dim))
        x[:] = np.NaN
        x[:lag+1, ] = x_init
        
        for i in np.arange(lag+1, x.shape[0]):
            x[i,] = np.sum(np.matmul(Phi, x[i-1:i-lag-1:-1][...,np.newaxis]), axis=(0, -1)) + \
                    np.random.multivariate_normal(np.repeat(0., dim), Sigma)
        x = x[101: ]
        return x

    def calculateSpecMatrix(self, f, inverse = False):
        specMat = []
        for coefs in self.coefs:
            matTemp = np.apply_along_axis(lambda f: self.calculateSpecMatrixHelper(f, coefs, sigma=self.sigma, inverse=inverse), axis=1, arr = f.reshape(-1,1))
            specMat.append(matTemp)
        S = [tf.linalg.LinearOperatorFullMatrix(si) for si in specMat]
        specMatFull = tf.linalg.LinearOperatorBlockDiag(S)
        Spec_mat_true = specMatFull.to_dense()
        return Spec_mat_true

    def calculateSpecMatrixHelper(self, f, Phi, sigma, inverse = False):
        # f:     a single frequency
        # Phi:   AR coefficient array. M-by-p-by-p. i-th row imply lag (M-i), p is dim of time series.
        dim = Phi.shape[1]
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        k = np.arange(1, Phi.shape[0]+1, 1)
        A_f_re = np.sum(Phi * np.cos(np.pi*2*k*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_im = - np.sum(Phi * np.sin(np.pi*2*k*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f = A_f_re + 1j * A_f_im
        dim = A_f.shape[0]
        A_bar_f = np.identity(dim) - A_f
        H_f = np.linalg.inv(A_bar_f)
        if inverse == False:
            Spec_mat = H_f @ Sigma @ H_f.conj().T
            return Spec_mat
        else:
            Spec_inv = A_bar_f.conj().T @ np.linalg.inv(Sigma) @ A_bar_f    
            return Spec_inv


if __name__ == '__main__':
    
    n = 640
    sigma = np.array([0.5])  
    with open('Data/var_coefs.pkl', 'rb') as handle:
            var_coefs = pickle.load(handle)

    Simulation = VarSim(var_coefs=var_coefs, n=n, sigma=sigma)
    x = Simulation.getData()
    freq = np.arange(1,np.floor_divide(1000, 2)+1, 1) / 1000
    Spec_mat_true = Simulation.calculateSpecMatrix(freq)

    for i in np.arange(Spec_mat_true.shape[1]):
        f, Pxx_den1 = signal.periodogram(x[:,i], fs=1)
        f = f[1:]
        Pxx_den1 = Pxx_den1[1:] / 2
        plt.scatter(f, np.log(Pxx_den1), marker = '.', s=1)
        plt.plot(freq, np.log(np.real(Spec_mat_true[:,i,i])))
        plt.show()
