
"""
find L1 errors given optimal lrs for map and elbo for VAR2 256 (method 1)
"""

import numpy as np
import pandas as pd

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import spec_vi_map

import modified_spec_vi_lr_elbo

import true_var # to simulate data

n = 256
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])

Simulation = true_var.VarmaSim(n=n)
freq = np.arange(1, int(n/2)+1) / n
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
spec_true = spec_true/(np.pi/0.5)

num_rep = 20
# load data
data_whole = pd.read_csv("C:/Users/jliu812/OneDrive - The University of Auckland/Desktop/PhD学习材料/L1L2_var2/var2_256_data.csv")
data_whole = data_whole.values

L1_VI_list = []
optimal_lr_map_list = []
optimal_lr_elbo_list = []

for i in range(num_rep):
    x = data_whole[i*n:((i+1)*n),:]

    # find optimal lrs for map and elbo
    lr_range = np.arange(0.0005, 0.0305, 0.0005)
    final_map = []
    lr_values = []

    for lr_value in lr_range:
        Spec = spec_vi_map.SpecVI(x) 
        lp = Spec.runModel(N_delta=30, N_theta=30, lr_map=lr_value, 
                           ntrain_map=10000, sparse_op=False, nchunks=1)
        
        final_map.append(lp[-1].numpy())
        lr_values.append(lr_value)

    max_map_index = np.argmax(final_map)
    optimal_lr_map = lr_range[max_map_index]
    
    Spec = modified_spec_vi_lr_elbo.SpecVI(x)
    result_list = Spec.runModel(N_delta=30, N_theta=30, lr_map=optimal_lr_map, 
                                ntrain_map=10000, sparse_op=False, nchunks=1)

    optimal_lr_elbo = result_list[-1]
    #find L1 error given optimal lrs for map and elbo-----------------------------------------------------

    spec_mat_median = result_list[0][1]/(np.pi/0.5)
    spec_mat_median = np.transpose(spec_mat_median, axes=(0, 2, 1))

    N1_VI = np.empty(n//2)
    for i in range(n//2):
        N1_VI[i] = np.sqrt(np.sum(np.diag((spec_mat_median[i,:,:]-spec_true[i,:,:]) @
                                    (spec_mat_median[i,:,:]-spec_true[i,:,:]))))

    L1_VI = np.mean(N1_VI)

    L1_VI_list.append(L1_VI)
    optimal_lr_map_list.append(optimal_lr_map)
    optimal_lr_elbo_list.append(optimal_lr_elbo)

result_df = pd.DataFrame({
    'L1_VI': L1_VI_list,
    'Optimal_LR_MAP': optimal_lr_map_list,
    'Optimal_LR_ELBO': optimal_lr_elbo_list
})

result_df.to_csv('optimal lrs and L1 by method 1.csv', index=False)






























