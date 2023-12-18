# -*- coding: utf-8 -*-
"""
Wind Data Analysis Demo Runbook

@author: Zhixiong Hu, UCSC
"""
#import os
#os.chdir('../code') # set "~/supplementary_materials/code" as work directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from moss.spec_vi import SpecVI # model module


HERE = os.path.dirname(os.path.abspath(__file__))


def test_demo():


    # load data
    my_data_frame = pd.read_csv(f'{HERE}/data/wind_detrend.csv', index_col=0, header=0)
    my_data = my_data_frame.values

    ## Visualize data ####################################
    df = my_data_frame
    color_ls = ['#4E79A7', '#E15759', '#499894', '#59A14F', '#D37295', "#9D7660"]
    ax = df.plot(subplots=True, figsize=(8,4), legend=False, layout=(df.shape[1],1), yticks=[0],
            sharex=True, sharey=False, color=color_ls, linewidth=1)
    for i in range(ax.shape[0]):
        if i != 0:
            ax[i][0].spines['top'].set_visible(False)
        if i != ax.shape[0]-1:
            ax[i][0].spines['bottom'].set_visible(False)
            ax[i][0].xaxis.set_ticks_position('none')
        if i == ax.shape[0]-1:
            xticklabels = ax[i][0].get_xticklabels()
            for e in xticklabels:
                e.set_text(e.get_text().split(' ')[0])
            ax[i][0].set_xticklabels(xticklabels)
            plt.setp(ax[i][0].get_xticklabels(), ha="center", rotation=00)
            ax[i][0].set_xlabel(r'Time Unit: Hour $\times$ 2', labelpad=10, fontsize='large')
            ax[i][0].tick_params(axis='both', labelsize='large')

        ax[i][0].margins(x=0) # remove margins from x-axis
        ax[i][0].set_ylim(df.min().min(), df.max().max())
        ax[i][0].set_yticklabels([df.columns[i]], color=color_ls[i], fontsize='x-large', fontweight='bold')
        # ax[i][0].set_ylabel("C"+str(i), rotation=0, labelpad=10)
    plt.tight_layout()
    plt.show()
    fig = ax[0][0].get_figure()

    ## Model run #######################################
    x = my_data
    Spec = SpecVI(x)
    '''
    If data_n-by_p_dim is large (e.g., for n>1000 & p>100) that throws OOM error, 
    change "sparse_op = True" to save memory by using sparse matrix operations. 
    
    When n and p is not too large, "sparse_op = False" (default) is recommended, 
    which works more light and faster!  
    '''
    # result_list = [spectral matrix estimates, model objects,]
    result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=1e-3, ntrain_map=8e3, sparse_op=False)
    spec_mat = result_list[0]
    freq = result_list[1].freq

    ## Result Visualization ###########################
    # Reproduce Figure 9
    fig, ax = plt.subplots(2,3, figsize = (11, 10))
    for ii in range(x.shape[1]):
        f, Pxx_den0 = signal.periodogram(x[:,ii], fs=1)
        f = f[1:]
        Pxx_den0 = Pxx_den0[1:] / 2
        f_CI = np.abs(spec_mat[...,ii,ii])

        ax[ii//3, ii%3].plot(f, np.log(Pxx_den0), marker = '.', markersize=2, linestyle = 'None')
        ax[ii//3, ii%3].plot(freq, np.log(f_CI[1]), linewidth=3, linestyle='--', color = 'black', label = 'Point Est.')
        ax[ii//3, ii%3].fill_between(freq, np.squeeze(np.log(f_CI[0])), np.squeeze(np.log(f_CI[2])),
                        color = 'darkgrey', alpha = 1, label = '95% CI')
        if my_data_frame.columns[ii] == 'EDU':
            ax[ii//3, ii%3].legend(fontsize='large', frameon=False)
        ax[ii//3, ii%3].tick_params(labelsize=15)
        ax[ii//3, ii%3].set_xlabel(r'$\nu$', fontsize=20, labelpad=10)
        #ax.set_ylabel(r'$\log$ spectral density', fontsize=20)
        ax[ii//3, ii%3].set_title(r'%s'%my_data_frame.columns[ii], pad=20, fontsize=20)
        ax[ii//3, ii%3].set_xlim([0, 0.5])
        ax[ii//3, ii%3].set_ylim([-12.5, 6.0])
        ax[ii//3, ii%3].grid(True)
    plt.tight_layout()
    plt.show()
    #fig.savefig('Wind19Analysis/fig/wind_f.eps')

    # Reproduce Figure 10 (subplots)
    for ii in np.arange(0,x.shape[1]):
        for jj in np.arange(ii+1, x.shape[1]):
            f_CI = np.square(spec_mat[...,ii,jj]) / np.abs(spec_mat[...,ii,ii]) / np.abs(spec_mat[...,jj,jj])
            fig, ax = plt.subplots(1,1, figsize = (5, 6), dpi=60)
            ax.plot(freq, np.squeeze(f_CI[1]), color = 'black', linewidth=3, linestyle = '-.', label = 'Point Est.')
            ax.fill_between(freq, f_CI[0], f_CI[2], color = 'darkgrey', alpha = 1, label = '95% CI')
            if my_data_frame.columns[ii] == 'SAC' and my_data_frame.columns[jj] == 'SMF':
                ax.legend(fontsize='x-large', frameon=False)
            ax.tick_params(labelsize=15)
            ax.set_xlabel(r'$\nu$', fontsize=20, labelpad=10)
            ax.set_xlim([0,0.5])
            ax.set_ylim([0., 1.])
            ax.grid(True)
            ax.set_title(r'%s vs %s'%(my_data_frame.columns[ii],my_data_frame.columns[jj]), pad=20, fontsize = 30)
            plt.show()
            #fig.savefig('../Wind19Analysis/fig/da_%s_%s.eps'%(my_data_frame.columns[ii], my_data_frame.columns[jj]))
