import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from scipy.fft import rfftfreq


def logistic(x):
    return 1/(1+np.exp(-x))

# TODO remove complex / dtype argument
def get_dcval(mean, J, dtype):
    if dtype == 'real':
        J = J*2
    elif dtype == 'complex':
        pass
    else:
        raise ValueError
    return mean*(J/(2*np.pi)) / 2

def conv_z_to_v(z, axis, dc=True):
    v = np.apply_along_axis(conv_z_to_v_vec, axis, z, dc)
    return v

def conv_v_to_z(v, axis, dc=True):
    z = np.apply_along_axis(conv_v_to_z_vec, axis, v, dc)
    return z

def conv_v_to_z_vec(v, dc=True):
    if dc is True:
        J = int((v.size - 1)/2)
        z = np.zeros(J + 1, dtype=complex)
        z[0] = v[0]
        v_temp = v[1:].reshape(-1,2) 
        temp_z = v_temp[:,0] + v_temp[:,1]*1j
        z[1:] = temp_z
    else: 
        J = int((v.size)/2)
        z = np.zeros(J, dtype=complex)
        v_temp = v.reshape(-1,2) 
        temp_z = v_temp[:,0] + v_temp[:,1]*1j
        z = temp_z
    return z

def conv_z_to_v_vec(z, dc=True):
    if dc is True:
        J = z.size - 1
        v = np.zeros(2*J + 1)
        v[0] = z[0].real
        v[1:] = np.array([z[1:].real, z[1:].imag]).T.reshape(J*2) 
    else:
        v = np.array([z.real, z.imag]).T.reshape(J*2) 
    return v

def conv_v_to_z_vec_old(v, dc=True):
    if dc is True:
        J = int((v.size - 1)/2)
        z = np.zeros(J + 1, dtype=complex)
        z[0] = 0.5*v[0] + 0*1j
        for j in range(1,J+1):
            a = v[2*j-1]
            b = v[2*j]
            c = conv_real_to_complex(a, b)
            z[j] = c
    else:
        J = int(v.size/2)
        z = np.zeros(J, dtype=complex)
        for j in range(J):
            a = v[2*j]
            b = v[2*j+1]
            c = conv_real_to_complex(a, b)
            z[j] = c
    return z

def conv_z_to_v_vec_old(z, dc=True):
    if dc is True:
        J = z.size - 1
        v = np.zeros(2*J + 1)
        v[0] = 2*z[0].real
        for j in range(1,J+1):
            a, b = conv_complex_to_real(z[j])
            v[2*j-1] = a
            v[2*j] = b
    else:
        J = z.size 
        v = np.zeros(2*J)
        for j in range(J):
            a, b = conv_complex_to_real(z[j])
            v[2*j] = a
            v[2*j+1] = b
    return v

def conv_complex_to_real(c):
    a = (c + np.conj(c)).real
    b = -(1j*(c - np.conj(c))).real
    return a, b

def conv_real_to_complex(a, b):
    c = 0.5*(a + 1j*b)
    return c

def get_freqs(time_sec, Fs):
    n = int(time_sec/(1/Fs))
    freqs = rfftfreq(n, d=1/Fs)
    return freqs[1:]

def add_zero(x_F, axis=1):
    zero = np.array([0 + 1j*0])
    n_trials = x_F.shape[0]
    zeros = np.repeat(zero, n_trials)
    new_x_F = np.concatenate([zeros[:,None], x_F], axis=axis)
    return new_x_F

def pickle_open(file):
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def pickle_save(data, save_name):
    with open(save_name, 'wb') as handle:
        pickle.dump(data, handle)


def draw_raster_single(spike_mat, trange, contrast=0.1, ax=None, ylabel=True,
                       origin='lower', **kwargs):
    ax = ax or plt.gca()
    contrast_scale = contrast
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'gray_r_a', [(0., 0., 0., 0.),  (0., 0., 0., 1.)])
    
    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('cmap', cmap)
    kwargs.setdefault('interpolation', None)
    kwargs.setdefault('extent', (trange[0], trange[-1], 0., spike_mat.shape[0]))
    
    ymin = 0
    ymax = spike_mat.shape[0]
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel('Spikes')
    ax.get_yaxis().set_ticks([])
    ax.tick_params(axis='x', which='major', labelsize=12)
    
    alpha = 0.2
    n_units = spike_mat.shape[0]
    ax.axhspan(0, n_units + 1, facecolor='white', alpha=alpha) #

    spikeraster = ax.imshow(spike_mat, origin=origin, **kwargs)
    spikeraster.set_clim(0., np.max(spike_mat) * contrast_scale)
    
    return spikeraster