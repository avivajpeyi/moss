import h5py
import numpy as np   
from scipy import signal 
import matplotlib.pyplot as plt
import modified_ll_128_spec_vi
from scipy.stats import median_abs_deviation
import time
start_time = time.time()

f = h5py.File('C://Users//jliu812//OneDrive - The University of Auckland//Desktop//gravitational wave data analysis//ADD_files//X_ETnoise_GP_uncorr.hdf5','r')
X_ETnoise_GP_uncorr = f['E1:STRAIN']

f = h5py.File('C://Users//jliu812//OneDrive - The University of Auckland//Desktop//gravitational wave data analysis//ADD_files//Y_ETnoise_GP_uncorr.hdf5','r')
Y_ETnoise_GP_uncorr = f['E2:STRAIN']

f = h5py.File('C://Users//jliu812//OneDrive - The University of Auckland//Desktop//gravitational wave data analysis//ADD_files//Z_ETnoise_GP_uncorr.hdf5','r')
Z_ETnoise_GP_uncorr = f['E3:STRAIN']

#----------------------------------------------------------------------------------------------

channels = np.column_stack((X_ETnoise_GP_uncorr, Y_ETnoise_GP_uncorr, Z_ETnoise_GP_uncorr))
channels_short = channels
q = 10**22/1.0

#result_list = Spec_m.runModel(N_delta=400, N_theta=400, lr_map=0.004, ntrain_map=8000, sparse_op=False, nchunks = nchunks)

time_interval = 2000
nchunks = 125
required_part = 128

Spec_m = modified_ll_128_spec_vi.SpecVI(channels_short * (q))
result_list = Spec_m.runModel(N_delta=500, N_theta=500, lr_map=0.003, ntrain_map=8000, 
                        sparse_op=False, nchunks = nchunks, time_interval = time_interval, required_part = required_part)


Ts = 1/ (channels.shape[0]/time_interval)
freq_original = np.fft.fftfreq(int(np.size(channels,0)/nchunks),Ts)

n = int(np.size(channels,0)/nchunks)
if np.mod(n, 2) == 0:
    # n is even
    freq_original = freq_original[0:int((n/2))]
else:
    # n is odd
    freq_original = freq_original[0:int((n-1)/2)]
       
total_len = channels.shape[0]
freq_range = total_len/time_interval/2
freq = freq_original[0:int(required_part/freq_range * freq_original.shape[0])]
    

spec_mat = result_list[0]
spec_mat_median = spec_mat[1] #shape (#freq, p, p) off diag elements are complex
spectral_density_all = result_list[3] #shape (500, #freq, p, p) off diag elements are complex
    
def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

n_samples, n_freq, p, _ = spectral_density_all.shape

real_spectral_density_all = np.zeros_like(spectral_density_all, dtype=float)
real_spec_mat_median = np.zeros_like(spec_mat_median, dtype=float)

for i in range(n_samples):
    for j in range(n_freq):
        real_spectral_density_all[i, j] = complex_to_real(spectral_density_all[i, j])

for j in range(n_freq):
    real_spec_mat_median[j] = complex_to_real(spec_mat_median[j])



mad = median_abs_deviation(real_spectral_density_all, axis=0, nan_policy='omit')
mad[mad == 0] = 1e-10 

def uniformmax_multi(mSample):
    N_sample, N, d, _ = mSample.shape
    C_help = np.zeros((N_sample, N, d, d))

    for j in range(N):
        for r in range(d):
            for s in range(d):
                C_help[:, j, r, s] = uniformmax_help(mSample[:, j, r, s])

    return np.max(C_help, axis=(1, 2, 3))

def uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)
    
max_std_abs_dev = uniformmax_multi(real_spectral_density_all) 
    
    
threshold = np.quantile(max_std_abs_dev, 0.9)    
lower_bound = real_spec_mat_median - threshold * mad
upper_bound = real_spec_mat_median + threshold * mad

def real_to_complex(matrix):
    n = matrix.shape[0]
    complex_matrix = np.zeros((n, n), dtype=complex)
    
    complex_matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)]
    
    complex_matrix[np.triu_indices(n, 1)] = matrix[np.triu_indices(n, 1)] - 1j * matrix[np.tril_indices(n, -1)]
    complex_matrix[np.tril_indices(n, -1)] = matrix[np.triu_indices(n, 1)] + 1j * matrix[np.tril_indices(n, -1)]
    
    return complex_matrix

complex_lower_bound = np.zeros_like(lower_bound, dtype=complex)
complex_upper_bound = np.zeros_like(upper_bound, dtype=complex)

for i in range(n_freq):
    complex_lower_bound[i] = real_to_complex(lower_bound[i])
    complex_upper_bound[i] = real_to_complex(upper_bound[i])

spec_mat_lower = complex_lower_bound
spec_mat_upper = complex_upper_bound


coh_all = result_list[4]
coh_med = result_list[2][1]

mad = median_abs_deviation(coh_all, axis=0, nan_policy='omit')
#mad[mad == 0] = 1e-10 

def uniformmax_multi(coh_whole):
    N_sample, N, d = coh_whole.shape
    C_help = np.zeros((N_sample, N, d))

    for j in range(N):
        for r in range(d):
                C_help[:, j, r] = uniformmax_help(coh_whole[:, j, r])

    return np.max(C_help, axis=(1, 2))

def uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)
    
max_std_abs_dev = uniformmax_multi(coh_all) 

threshold = np.quantile(max_std_abs_dev, 0.9)    
coh_lower = coh_med - threshold * mad
coh_upper = coh_med + threshold * mad

#-------------------------------------------------------------------------------------------------------
#plots for psd matrix
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
from matplotlib.lines import Line2D

for i in range(3):
    for j in range(3):
        if i == j:
            f, Pxx_den0 = signal.periodogram(channels[:,i], fs=channels.shape[0]/time_interval)
            f = f[1:]
            Pxx_den0 = Pxx_den0[1:] / 2
            axes[i, j].plot(f, Pxx_den0, marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].plot(freq, spec_mat_median[..., i, i]/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, spec_mat_lower[..., i, i]/(q)**2/(freq_original[-1]/0.5),
                            spec_mat_upper[..., i, i]/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            axes[i, j].text(0.95, 0.95, r'$f_{{{}, {}}}$'.format(i+1, i+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
            axes[i, j].set_ylim([10**(-52), 10**(-46)])
            axes[i, j].set_yscale('log')
            axes[i, j].axvline(x=10, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(10, 10**(-52)-0.023, '10', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=50, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(50, 10**(-52)-0.023, '50', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=90, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(90, 10**(-52)-0.023, '90', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            
        elif i < j:  
            
            axes[i, j].plot(freq, np.real(spec_mat_median[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, np.real(spec_mat_lower[..., i, j])/(q)**2/(freq_original[-1]/0.5), 
                                    np.real(spec_mat_upper[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            y = np.apply_along_axis(np.fft.fft, 0, channels)
            n = channels.shape[0]
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.real(cross_spectrum_fij)/(freq_original[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].text(0.95, 0.95, r'$\Re(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
            axes[i, j].axvline(x=10, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(10, 10**(-52)-0.023, '10', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=50, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(50, 10**(-52)-0.023, '50', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=90, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(90, 10**(-52)-0.023, '90', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
        
            axes[i, j].set_yscale('symlog', linthresh=1e-49)
        else:
            
            axes[i, j].plot(freq, -np.imag(spec_mat_median[..., i, j])/(q)**2/(freq_original[-1]/0.5), linewidth=1, color='green', linestyle="-")
            axes[i, j].fill_between(freq, -np.imag(spec_mat_lower[..., i, j])/(q)**2/(freq_original[-1]/0.5),
                                    -np.imag(spec_mat_upper[..., i, j])/(q)**2/(freq_original[-1]/0.5), color='lightgreen', alpha=1)
            
            y = np.apply_along_axis(np.fft.fft, 0, channels)
            n = channels.shape[0]
            if np.mod(n, 2) == 0:
                # n is even
                y = y[0:int(n/2)]
            else:
                # n is odd
                y = y[0:int((n-1)/2)]
            y = y / np.sqrt(n)
            #y = y[1:]
            cross_spectrum_fij = y[:, i] * np.conj(y[:, j])
            
            axes[i, j].plot(f, np.imag(cross_spectrum_fij)/(freq_original[-1]/0.5),
                            marker='', markersize=0, linestyle='-', color='lightgray', alpha=0.3)
            
            axes[i, j].text(0.95, 0.95, r'$\Im(f_{{{}, {}}})$'.format(i+1, j+1), transform=axes[i, j].transAxes, 
                            horizontalalignment='right', verticalalignment='top', fontsize=14)

            axes[i, j].set_xlim([5, 128])
            axes[i, j].axvline(x=10, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(10, 10**(-52)-0.023, '10', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=50, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(50, 10**(-52)-0.023, '50', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].axvline(x=90, color='red', linestyle='--', linewidth=0.5)
            axes[i, j].text(90, 10**(-52)-0.023, '90', color='black', ha='center', va='top', fontsize=10, transform=axes[i, j].get_xaxis_transform())
            
            axes[i, j].set_yscale('symlog', linthresh=1e-49)
fig.text(0.5, 0.1, 'Frequency [Hz]', ha='center', va='center', fontsize=20)
fig.text(0.08, 0.5, 'Strain PSD [1/Hz]', ha='center', va='center', rotation='vertical', fontsize=20)

fig.legend(handles=[Line2D([], [], color='lightgray', label='Periodogram'),
                Line2D([], [], color='green', label='Estimated PSD'),
            Line2D([], [], color='lightgreen', label='90% CI')],
             loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=14)
       


# squared coherence-------------------------------------------------------------------------

fig, ax = plt.subplots(1,1, figsize = (20, 6))
plt.xlim([5, 128])
plt.plot(freq, np.squeeze(coh_med[:,0]), color = 'green', linestyle="-", label = 'coherence for X Y')
plt.fill_between(freq, np.squeeze(coh_lower[:,0]), np.squeeze(coh_upper[:,0]),
                    color = ['lightgreen'], alpha = 1, label = '90% CI for X Y')


plt.plot(freq, np.squeeze(coh_med[:,1]), color = 'blue', linestyle="-", label = 'coherence for X Z')
plt.fill_between(freq, np.squeeze(coh_lower[:,1]), np.squeeze(coh_upper[:,1]),
                    color = ['lightblue'], alpha = 1, label = '90% CI for X Z')


plt.plot(freq, np.squeeze(coh_med[:,2]), color = 'red', linestyle="-", label = 'coherence for Y Z')
plt.fill_between(freq, np.squeeze(coh_lower[:,2]), np.squeeze(coh_upper[:,2]),
                    color = ['lightcoral'], alpha = 1, label = '90% CI for Y Z')

plt.xlabel('Frequency [Hz]', fontsize=20, labelpad=10)   
plt.ylabel('Squared Coherency', fontsize=20, labelpad=10)   
plt.title('Squared coherence for ET noise with uncorrelated GP', pad=20, fontsize = 20)

plt.legend(loc='upper left', fontsize='medium')
plt.ylim([0, 0.7])
plt.axvline(x=10, color='red', linestyle='--', linewidth=0.5)
ax.text(10, -0.023, '10', color='black', ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform())

plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5)
ax.text(50, -0.023, '50', color='black', ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform())

plt.axvline(x=90, color='red', linestyle='--', linewidth=0.5)
ax.text(90, -0.023, '90', color='black', ha='center', va='top', fontsize=10, transform=ax.get_xaxis_transform())


plt.grid(True)

#---------------------------------------------------------------------------------------------

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.2f} seconds")





























