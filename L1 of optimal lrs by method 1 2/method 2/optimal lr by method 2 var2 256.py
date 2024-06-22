
"""
find L1 errors given optimal lrs for map and elbo for VAR2 256 (method 2)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import modified_lr_spec_vi_elbo

from data import true_var

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
optimal_lr_elbo_list = []

for i in range(num_rep):
    x = data_whole[i*n:((i+1)*n),:]

    # find optimal lrs for map and elbo
    lr_range = np.arange(0.0005, 0.0205, 0.0005)
    final_elbo = []
    lr_values = []
    all_samp = []
    N_delta = 30
    N_theta = 30
    
    for lr_value in lr_range:
        Spec = modified_lr_spec_vi_elbo.SpecVI(x) 
        result_list = Spec.runModel(N_delta=N_delta, N_theta=N_theta, lr_map=lr_value, 
                                    ntrain_map=10000, sparse_op=False, nchunks=1)
        
        losses = result_list[0]
        samp = result_list[2]
        final_elbo.append(-losses[-1].numpy())
        all_samp.append(samp)
        lr_values.append(lr_value)

    max_elbo_index = np.argmax(final_elbo)
    optimal_lr = lr_range[max_elbo_index]
    best_samp = all_samp[max_elbo_index]
    
    #find estimated psd given the max elbo-----------------------------------------------------------------
    Xmat_delta, Xmat_theta = result_list[1].Xmtrix(N_delta=N_delta, N_theta=N_theta)
    
    delta2_all_s = tf.exp(tf.matmul(Xmat_delta, tf.transpose(best_samp[0], [0,2,1]))) #(500, #freq, p)

    theta_re_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[2], [0,2,1])) #(500, #freq, p(p-1)/2)
    theta_im_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[4], [0,2,1]))
    
    theta_all_s = -(tf.complex(theta_re_s, theta_im_s)) #(500, #freq, p(p-1)/2)
    theta_all_np = theta_all_s.numpy() 

    D_all = tf.map_fn(lambda x: tf.linalg.diag(x), delta2_all_s).numpy() #(500, #freq, p, p)


    num_slices, num_freq, num_elements = theta_all_np.shape
    p_dim = result_list[1].p_dim
    row_indices, col_indices = np.tril_indices(p_dim, k=-1)
    diag_matrix = np.eye(p_dim, dtype=np.complex64)
    T_all = np.tile(diag_matrix, (num_slices, num_freq, 1, 1))
    T_all[:, :, row_indices, col_indices] = theta_all_np.reshape(num_slices, num_freq, -1)

    T_all_conj_trans = np.conj(np.transpose(T_all, axes=(0, 1, 3, 2)))
    
    D_all_inv = np.linalg.inv(D_all)
    
    spectral_density_inverse_all = T_all_conj_trans @ D_all_inv @ T_all
    spectral_density_all = np.linalg.inv(spectral_density_inverse_all) 

    num_freq = spectral_density_all.shape[1]
    spectral_density_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)
    
    diag_indices = np.diag_indices(p_dim)
    spectral_density_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(spectral_density_all[:, :, diag_indices[0], diag_indices[1]], [0.025, 0.5, 0.975], axis=0)
    
    triu_indices = np.triu_indices(p_dim, k=1)
    real_part = (np.real(spectral_density_all[:, :, triu_indices[0], triu_indices[1]]))
    imag_part = (np.imag(spectral_density_all[:, :, triu_indices[0], triu_indices[1]]))
    
    spectral_density_q[0, :, triu_indices[0], triu_indices[1]] = (np.quantile(real_part, 0.025, axis=0) - 1j * np.quantile(imag_part, 0.025, axis=0)).T
    spectral_density_q[1, :, triu_indices[0], triu_indices[1]] = (np.quantile(real_part, 0.50, axis=0) - 1j * np.quantile(imag_part, 0.50, axis=0)).T
    spectral_density_q[2, :, triu_indices[0], triu_indices[1]] = (np.quantile(real_part, 0.975, axis=0) - 1j * np.quantile(imag_part, 0.975, axis=0)).T
    
    spectral_density_q[:, :, triu_indices[1], triu_indices[0]] = np.conj(spectral_density_q[:, :, triu_indices[0], triu_indices[1]])
    
    spec_mat_median = spectral_density_q[1]/(np.pi/0.5)
    spec_mat_median = np.transpose(spec_mat_median, axes=(0, 2, 1))
    
    #find L1 error given optimal lrs in phase that maximise elbo-------------------------------------------
    N1_VI = np.empty(n//2)
    for i in range(n//2):
        N1_VI[i] = np.sqrt(np.sum(np.diag((spec_mat_median[i,:,:]-spec_true[i,:,:]) @
                                    (spec_mat_median[i,:,:]-spec_true[i,:,:]))))

    L1_VI = np.mean(N1_VI)
    
    L1_VI_list.append(L1_VI)
    optimal_lr_elbo_list.append(optimal_lr)
    
result_df = pd.DataFrame({
    'L1_VI': L1_VI_list,
    'Optimal_LR_ELBO': optimal_lr_elbo_list
})

result_df.to_csv('optimal lrs and L1 by method 2.csv', index=False)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 