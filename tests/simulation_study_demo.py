# -*- coding: utf-8 -*-
"""
Multivariate Spectral Analysis Simulation Study Demo Runbook

@author: Zhixiong Hu
"""

#import os
#os.chdir('../code') # set "~/supplementary_materials/code" as work directory



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from model import spec_vi # model module
from data import var_sim, varma_sim # to simulate data

#########################################################
## Simulation Study 1: Bivariate Analysis ###############

## simulate data ##################################
np.random.seed(1234567)
n = 1024
sigma = np.array([[1., 0.8], [0.8, 1.]])  
var_coef = np.array([[[0.2, 0.5], [0., -0.2]], [[0., 0.], [0.5, -0.2]]])
vma_coef = np.array([[[1.,0.],[0.,1.]], [[0.6, 0], [0.2, -0.5]], 
                    [[0.3, 0], [0, 0.3]]])

    
Simulation = varma_sim.VarmaSim(n=n)
x = Simulation.simData(var_coef, vma_coef, sigma=sigma)
freq = np.arange(1,np.floor_divide(500*2, 2)+1, 1) / (500*2)
spec_true = Simulation.calculateSpecMatrix(freq, var_coef, vma_coef, sigma)


## Model run #######################################
Spec = spec_vi.SpecVI(x)
'''
If data_n-by_p_dim is large (e.g., for n>1000 & p>100) to throw OOM warning, 
change "sparse_op = True" to save memory by using sparse matrix operations. 

When n and p is not too large, "sparse_op = False" (default) is recommended, 
which works lighter and faster!  
'''
# result_list = [spectral matrix estimates, model objects,]
result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, sparse_op=False)
spec_mat = result_list[0]
freq = result_list[1].freq

## Result Visualization ###########################
fig, ax = plt.subplots(1,3, figsize = (11, 5))
for i in range(2):
    f, Pxx_den0 = signal.periodogram(x[:,i], fs=1)
    f = f[1:]
    Pxx_den0 = Pxx_den0[1:] / 2
    f_CI = np.log(np.abs(spec_mat[...,i,i]))
    ax[i].plot(f, np.log(Pxx_den0), marker = '.', markersize=2, linestyle = 'None')
    ax[i].plot(freq, np.log(np.real(spec_true[:,i,i])), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[i].fill_between(freq, np.squeeze(f_CI[0]), np.squeeze(f_CI[2]),
                color = 'darkgray', alpha = 1, label = '95% CI')
    ax[i].tick_params(labelsize=15)
    ax[i].set_xlabel(r'$\nu$', fontsize=20, labelpad=10)
    ax[i].set_title(r'$\log f_{%s,%s}$'%(i+1, i+1), pad=20, fontsize=20)
    ax[i].set_xlim([0, 0.5])
    ax[i].grid(True)
f_CI = np.square(spec_mat[...,0,1]) / np.abs(spec_mat[...,0,0]) / np.abs(spec_mat[...,1,1])
ax[2].plot(freq, np.absolute(spec_true[:,0,1])**2 / (np.real(spec_true[:,0,0] * np.real(spec_true[:,1,1]))), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
ax[2].fill_between(freq, np.squeeze(f_CI[0]), np.squeeze(f_CI[2]), 
                   color = 'darkgrey', alpha = 1, label = '95% CI')
ax[2].set_xlim([0,0.5])
ax[2].set_ylim([0., 1.])
ax[2].set_xlabel(r'$\nu$', fontsize=20, labelpad=10)    
ax[2].set_title(r'$\rho^2_{1,2}$', pad=20, fontsize = 20)
ax[2].grid(True)
plt.tight_layout()
plt.show()



#########################################################
## Simulation Study 2: High-dimensional Analysis ########
import pickle
## simulate data ##################################
n = 640; sigma = 1.
with open('Data/var_coefs.pkl', 'rb') as handle:
    var_coefs = pickle.load(handle)
Simulation = var_sim.VarSim(var_coefs=var_coefs, n=n, sigma=sigma)
np.random.seed(1234567)
x = Simulation.getData()
freq = np.arange(1,np.floor_divide(1000, 2)+1, 1) / 1000
Spec_mat_true = Simulation.calculateSpecMatrix(freq)

## Model run ############################################
# here to save Demo time, run the first 8 dimensions as a demo.
p=8 # change to 21 if computer is fast enough.
Spec = spec_vi.SpecVI(x[:,:p])
result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=5e-4, ntrain_map=5e3, sparse_op=False)
spec_mat = result_list[0]

## Result Visualization ###########################
# Reproduce Figure 4
fig = plt.figure(figsize=(10, 6.5), dpi= 60, facecolor='w', edgecolor='k')
#fig.suptitle('Spectral Density Estimates', fontsize='x-large', y=1.05)
fig.subplots_adjust(top=0.95)
for idx in range(8):
    f, Pxx_den0 = signal.periodogram(x[:,idx], fs=1)
    f = f[1:]
    Pxx_den0 = Pxx_den0[1:] / 2
    f_CI = np.abs(spec_mat[...,idx,idx])
    ax = fig.add_subplot(2,4, idx+1)
    ax.plot(f, Pxx_den0, marker = '.', markersize=1, linestyle = 'None')
    ax.plot(freq, np.real(Spec_mat_true[:,idx,idx]), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax.fill_between(freq, np.squeeze(f_CI[0]), np.squeeze(f_CI[2]),
                    color = 'darkgray', alpha = 1, label = '95% CI')
    #if idx == 0:
    #    ax.legend(fontsize=17, frameon=False)
    ax.set_yscale('log')
    ax.tick_params(labelsize='small')
    ax.set_xlabel(r'$\nu$', fontsize='large')
    ax.set_title(r'$\log f_{%d, %d}$ '%(idx+1, idx+1), fontsize="x-large", pad=12)
    ax.set_xlim([0, 0.5])
    ax.grid(True)
    #ax.get_yaxis().set_ticks([])
plt.tight_layout()
plt.show()

# Reproduce Figure 5 (if the computer is fast enough to run a total 21-dim analysis)
# =============================================================================
# fig = plt.figure(figsize=(10, 6.5), dpi= 60, facecolor='w', edgecolor='k')
# fig.suptitle('Squared Coherence Estimates (part)', fontsize='x-large', y=1.05)
# fig.subplots_adjust(top=0.95)
# idx = 0
# for ii in [ 8, 9, 10, 15,]:
#     if idx == 8:
#         break
#     for jj in np.arange(ii+1, ii+3):
#         f_CI = np.square(spec_mat[...,ii,jj]) / np.abs(spec_mat[...,ii,ii]) / np.abs(spec_mat[...,jj,jj])
#         ax = fig.add_subplot(2,4, idx+1)
#         ax.plot(freq, np.absolute(Spec_mat_true[:, ii, jj])**2 / (np.real(Spec_mat_true[:,ii,ii] * np.real(Spec_mat_true[:,jj,jj]))) + 7e-3, linestyle='-.', 
#                 color = 'red', linewidth=2, label = 'Truth')
#         ax.fill_between(freq, np.squeeze(f_CI[0])+7e-3, np.squeeze(f_CI[2])+7e-3, 
#                         color = 'darkgrey', alpha = 1, label = '95% CI')
#         #if ii == 0 and jj == 1:
#         #    ax.legend(fontsize=17, frameon=False)
#         ax.tick_params(labelsize='small')
#         ax.set_xlabel(r'$\nu$', fontsize='large')
#         ax.set_xlim([0,0.5])
#         ax.set_ylim([0., 1.])
#         ax.grid(True)
#         ax.set_title(r'$\rho_{%d,%d}^{2}$'%(ii+1,jj+1), fontsize = 'x-large', pad=12)
#         idx += 1
# plt.tight_layout()
# plt.show()
# =============================================================================
