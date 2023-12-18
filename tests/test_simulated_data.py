import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
from moss.spec_vi import SpecVI



def test_bivariate(Simulation):
    ## simulate data ##################################
    x, freq, spec_true = Simulation

    ## Model run #######################################
    Spec = SpecVI(x)
    result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, sparse_op=False)
    _plot_results(x, spec_true, spec_mat=result_list[0], freq=result_list[1].freq, label='ORIGINAL')


def _plot_results(x, spec_true, spec_mat, freq, label):
    ## Result Visualization ###########################
    fig, ax = plt.subplots(1, 3, figsize=(11, 5))
    for i in range(2):
        f, Pxx_den0 = signal.periodogram(x[:, i], fs=1)
        f = f[1:]
        Pxx_den0 = Pxx_den0[1:] / 2
        f_CI = np.log(np.abs(spec_mat[..., i, i]))
        ax[i].plot(f, np.log(Pxx_den0), marker='.', markersize=2, linestyle='None')
        ax[i].plot(freq, np.log(np.real(spec_true[:, i, i])), linewidth=2, color='red', linestyle="-.", label='Truth')
        ax[i].fill_between(freq, np.squeeze(f_CI[0]), np.squeeze(f_CI[2]),
                           color='darkgray', alpha=1, label='95% CI')
        ax[i].tick_params(labelsize=15)
        ax[i].set_xlabel(r'$\nu$', fontsize=20, labelpad=10)
        ax[i].set_title(r'$\log f_{%s,%s}$ ' % (i + 1, i + 1), pad=20, fontsize=20)
        ax[i].set_xlim([0, 0.5])
        ax[i].grid(True)
    f_CI = np.square(spec_mat[..., 0, 1]) / np.abs(spec_mat[..., 0, 0]) / np.abs(spec_mat[..., 1, 1])
    ax[2].plot(freq, np.absolute(spec_true[:, 0, 1]) ** 2 / (np.real(spec_true[:, 0, 0] * np.real(spec_true[:, 1, 1]))),
               linewidth=2, color='red', linestyle="-.", label='Truth')
    ax[2].fill_between(freq, np.squeeze(f_CI[0]), np.squeeze(f_CI[2]),
                       color='darkgrey', alpha=1, label='95% CI')
    ax[2].set_xlim([0, 0.5])
    ax[2].set_ylim([0., 1.])
    ax[2].set_xlabel(r'$\nu$', fontsize=20, labelpad=10)
    ax[2].set_title(r'$\rho^2_{1,2}$', pad=20, fontsize=20)
    ax[2].grid(True)
    plt.tight_layout()
    plt.show()
