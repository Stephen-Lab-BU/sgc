import numpy as np
from scipy.fft import rfftfreq

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