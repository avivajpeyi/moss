"""
find mu (means) for each component of the multivariate time series

"""
import os
os.chdir('C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/PhD学习材料/supplementary_materials/code') # set "~/supplementary_materials/code" as work directory

import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import modified_lnl_spec_vi_mu # model module
import true_var2_mu # to simulate data

n = 512
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])
 
Simulation = true_var2_mu.VarmaSim(n=n)
freq = (np.arange(1,np.floor_divide(n, 2)+1, 1) / (n))
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
x = Simulation.simData(varCoef, vmaCoef, sigma=sigma)

nchunks = 1
Spec = modified_lnl_spec_vi_mu.SpecVI(x)
result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=0.01, ntrain_map=6000, sparse_op=False, nchunks = nchunks)

mu_all = np.squeeze(result_list[-1])
mu = np.mean(mu_all, axis=0)

print('The mean for each component of the multivariate time series are', mu)









































