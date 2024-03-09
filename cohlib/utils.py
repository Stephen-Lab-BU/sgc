import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from scipy.fft import rfftfreq


def logistic(x):
    return 1/(1+np.exp(-x))

def get_dcval(mean, J):
    return mean*(J/(2*np.pi))

def conv_z_to_v(z, axis, dc=True):
    v = np.apply_along_axis(conv_z_to_v_vec, axis, z)
    return v

def conv_z_to_v_vec(z, dc=True):
    if dc is True:
        J = z.size - 1
        v = np.zeros(2*J + 1)
        v[0] = z[0].real
        for j in range(1,J):
            a, b = conv_complex_to_real(z[j])
            v[2*j+1] = a
            v[2*j+2] = b
    else:
        J = z.size 
        v = np.zeros(2*J + 1)
        v[0] = z[0].real
        for j in range(0,J):
            a, b = conv_complex_to_real(z[j])
            v[2*j+1] = a
            v[2*j+2] = b
    return v

def conv_complex_to_real(c):
    a = (c + np.conj(c)).real
    b = (1j*(c - np.conj(c))).real
    return a, b

def conv_real_to_complex(a, b):
    c = 0.5*(a - 1j*b)
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