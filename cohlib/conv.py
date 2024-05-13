import numpy as np

# Conversion between complex and real representation of Fourier coefs.


# TODO test
def conv_z_to_v(z, axis, dc=True):
    """
    Convert complex-valued fourier coefs to real-valued.
    Args:
        z (complex array): ndarray of complex fourier coefs
        axis (int): frequency axis
        dc (bool): True if zeroth entry is DC term
    """
    v = np.apply_along_axis(conv_z_to_v_vec, axis, z, dc)
    return v


# TODO test
def conv_v_to_z(v, axis, dc=True):
    """
    Convert complex-valued fourier coefs to real-valued.
    Args:
        v (complex array): ndarray of real fourier coefs
        axis (int): frequency axis
        dc (bool): True if zeroth entry is DC term
    """
    z = np.apply_along_axis(conv_v_to_z_vec, axis, v, dc)
    return z


# TODO test
def conv_z_to_v_vec(z, dc=True):
    """
    Convert vector of complex-valued fourier coefs to complex-valued.
    """
    if dc is True:
        J = z.size - 1
        v = np.zeros(2 * J + 1)
        v[0] = z[0].real
        v[1:] = np.array([z[1:].real, z[1:].imag]).T.reshape(J * 2)
    else:
        J = z.size
        v = np.array([z.real, z.imag]).T.reshape(J * 2)
    return v


# TODO test
def conv_v_to_z_vec(v, dc=True):
    """
    Convert vector of real-valued fourier coefs to real-valued.
    """
    if dc is True:
        J = int((v.size - 1) / 2)
        z = np.zeros(J + 1, dtype=complex)
        z[0] = v[0]
        v_temp = v[1:].reshape(-1, 2)
        temp_z = v_temp[:, 0] + v_temp[:, 1] * 1j
        z[1:] = temp_z
    else:
        J = int((v.size) / 2)
        z = np.zeros(J, dtype=complex)
        v_temp = v.reshape(-1, 2)
        temp_z = v_temp[:, 0] + v_temp[:, 1] * 1j
        z = temp_z
    return z


def transform_cov_c2r(complex_cov):
    """
    Convert covariance matrix from complex-valued to real-valued.
    """
    dim = complex_cov.shape[0]
    A = np.real(complex_cov)
    B = np.imag(complex_cov)
    rcov = np.zeros((2 * dim, 2 * dim))

    rcov = np.block([[A, -B], [B, A]])

    return rcov / 2


def transform_cov_r2c(real_cov):
    """
    Convert covariance matrix from real-valued to complex-valued.
    """
    dimR = real_cov.shape[0]
    dimC = int(dimR / 2)
    ccov = np.zeros((dimC, dimC), dtype=complex)

    A = real_cov[:dimC, :dimC]
    B = real_cov[:dimC, dimC:]
    C = real_cov[dimC:, :dimC]
    D = real_cov[dimC:, dimC:]

    ccov = (A + D) + 1j * (C - B)

    return ccov


def rearrange_mat(mat, K):
    """
    Rearrange real-valued covariance matrix order from frequency block to group block.
    """
    temp = np.tile(np.array([1, 0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1, 1, axis=1)
    f3 = np.roll(f1, 1, axis=0)
    f4 = np.roll(f1, 1, axis=(0, 1))

    A = mat[f1.astype(bool)].reshape(K, -1)
    B = mat[f2.astype(bool)].reshape(K, -1)
    C = mat[f3.astype(bool)].reshape(K, -1)
    D = mat[f4.astype(bool)].reshape(K, -1)

    new_mat = np.block([[A, B], [C, D]])
    return new_mat


def reverse_rearrange_mat(mat, K):
    """
    Rearrange real-valued covariance matrix order from group block to frequency block.
    """
    temp = np.tile(np.array([1, 0]), K)
    f1 = np.outer(temp, temp)
    f2 = np.roll(f1, 1, axis=1)
    f3 = np.roll(f1, 1, axis=0)
    f4 = np.roll(f1, 1, axis=(0, 1))

    dimC = K

    A = mat[:dimC, :dimC]
    B = mat[:dimC, dimC:]
    C = mat[dimC:, :dimC]
    D = mat[dimC:, dimC:]

    new_mat = np.zeros_like(mat)

    new_mat[f1.astype(bool)] = A.flatten()
    new_mat[f2.astype(bool)] = B.flatten()
    new_mat[f3.astype(bool)] = C.flatten()
    new_mat[f4.astype(bool)] = D.flatten()

    return new_mat
