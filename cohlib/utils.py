import random
import numpy as np
import pickle
from scipy.fft import rfftfreq


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    return seed

def logistic(x):
    return 1/(1+np.exp(-x))

# TODO test
def get_dcval(mean, J):
    Jv = J*2
    return mean*(Jv/(2*np.pi)) / 2

# TODO test
def conv_z_to_v(z, axis, dc=True):
    v = np.apply_along_axis(conv_z_to_v_vec, axis, z, dc)
    return v

# TODO test
def conv_v_to_z(v, axis, dc=True):
    z = np.apply_along_axis(conv_v_to_z_vec, axis, v, dc)
    return z

# TODO test
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

# TODO test
def conv_z_to_v_vec(z, dc=True):
    if dc is True:
        J = z.size - 1
        v = np.zeros(2*J + 1)
        v[0] = z[0].real
        v[1:] = np.array([z[1:].real, z[1:].imag]).T.reshape(J*2) 
    else:
        J = z.size 
        v = np.array([z.real, z.imag]).T.reshape(J*2) 
    return v


# TODO test
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


def transform_cov_c2r(complex_cov):
    dim = complex_cov.shape[0]
    A = np.real(complex_cov)
    B = np.imag(complex_cov)
    rcov = np.zeros((2*dim, 2*dim))
    
    rcov = np.block([[A, -B],
                     [B, A]])

    return rcov/2


def rearrange_mat(mat, K):
    temp = np.tile(np.array([1,0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1,1,axis=1)
    f3 = np.roll(f1,1,axis=0)
    f4 = np.roll(f1,1,axis=(0,1))

    A = mat[f1.astype(bool)].reshape(K,-1)
    B = mat[f2.astype(bool)].reshape(K,-1)
    C = mat[f3.astype(bool)].reshape(K,-1)
    D = mat[f4.astype(bool)].reshape(K,-1)
    
    new_mat = np.block([[A, B],
                       [C, D]])
    return new_mat

def reverse_rearrange_mat(mat, K):
    temp = np.tile(np.array([1,0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1,1,axis=1)
    f3 = np.roll(f1,1,axis=0)
    f4 = np.roll(f1,1,axis=(0,1))

    dimC = K

    A = mat[:dimC,:dimC]
    B = mat[:dimC,dimC:]
    C = mat[dimC:,:dimC]
    D = mat[dimC:,dimC:]

    new_mat = np.zeros_like(mat)

    new_mat[f1.astype(bool)] = A.flatten()
    new_mat[f2.astype(bool)] = B.flatten()
    new_mat[f3.astype(bool)] = C.flatten()
    new_mat[f4.astype(bool)] = D.flatten()
    
    return new_mat

def est_cov_r2c(real_cov):
    dimR = real_cov.shape[0]
    dimC = int(dimR/2)
    ccov = np.zeros((dimC, dimC), dtype=complex)

    A = real_cov[:dimC,:dimC]
    B = real_cov[:dimC,dimC:]
    C = real_cov[dimC:,:dimC]
    D = real_cov[dimC:,dimC:]

    ccov = (A + D) + 1j*(C - B)

    return ccov

transform_cov_r2c = est_cov_r2c


# UNUSED
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