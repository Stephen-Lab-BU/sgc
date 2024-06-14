import numpy as np


# TODO add option for DC that applies associated normalization correctly
def construct_real_idft_mod(slen, J, J_max, fs, dc=False, order='standard'):
    Jv = int(J * 2) + 2
    Jv_max = int(J_max * 2) + 2

    Wv = np.zeros((slen, Jv_max))
    for t in range(slen):
        if dc is True:
            for j in range(J_max + 1):
                Wv[t, 2 * j] = np.cos((t + 1) * np.pi * j / (J + 1))
                Wv[t, 2 * j + 1] = -np.sin(((t + 1) * np.pi * j) / (J + 1))
        else:
            for j in range(1, J_max + 1):
                Wv[t, 2 * (j - 1)] = np.cos((t + 1) * np.pi * j / (J + 1))
                Wv[t, 2 * (j - 1) + 1] = -np.sin((t + 1) * np.pi * j / (J + 1))
    Wv = Wv * ((2 * np.pi) / (J + 1))
    if dc is True:
        Wv = np.delete(Wv, 1, 1)

    if order == "standard":
        pass
    elif order == "real-imag":
        if dc is True:
            raise NotImplementedError
        else:
            re_filt = np.arange(Wv.shape[1]) % 2 == 0
            im_filt = np.arange(Wv.shape[1]) % 2 == 1
            Wv_re = Wv[:, re_filt]
            Wv_im = Wv[:, im_filt]
            Wv_ri = np.concatenate([Wv_re, Wv_im], axis=1)
            Wv = Wv_ri
    else:
        raise ValueError
    return Wv


def construct_real_idft(slen, J, fs, dc=False, order="standard"):
    if dc is True:
        Jv = int(J * 2) + 2
    else:
        Jv = int(J * 2)
    Wv = np.zeros((slen, Jv))

    for t in range(slen):
        if dc is True:
            for j in range(J + 1):
                Wv[t, 2 * j] = np.cos((t + 1) * np.pi * j / (J + 1))
                Wv[t, 2 * j + 1] = -np.cos((t + 1) * np.pi * j / (J + 1))
        else:
            for j in range(1, J + 1):
                Wv[t, 2 * (j - 1)] = np.cos((t + 1) * np.pi * j / (J + 1))
                Wv[t, 2 * (j - 1) + 1] = -np.sin((t + 1) * np.pi * j / (J + 1))
    Wv = Wv * ((2 * np.pi) / (J + 1))

    if dc is True:
        Wv = np.delete(Wv, 1, 1)

    if order == "standard":
        pass
    elif order == "real-imag":
        if dc is True:
            raise NotImplementedError
        else:
            re_filt = np.arange(Wv.shape[1]) % 2 == 0
            im_filt = np.arange(Wv.shape[1]) % 2 == 1
            Wv_re = Wv[:, re_filt]
            Wv_im = Wv[:, im_filt]
            Wv_ri = np.concatenate([Wv_re, Wv_im], axis=1)
            Wv = Wv_ri
    else:
        raise ValueError

    return Jv


def generate_harmonic_dict(slen, fs, res, frange, widx=0):
    """

    Generate dictionary with harmonic basis (not necessarily orthogonal)

    Inputs
    ======
    slen: signal length
    fs: Sampling frequency
    res: desired spectral resolution
    frange: array_like. Frequency range
    widx: Window index. Defaults to 0

    Outputs
    =======
    d: dictionary
    oflag: indicator whether the dictionary is orthogonal or not
    """

    assert frange[0] >= 0 and frange[1] >= 0, "Non-negative frequency required"
    assert frange[1] <= fs, "Has to be lower than the sampling frequency"

    n_elements = int(fs / res)

    k_low = int(np.ceil(frange[0] / fs * n_elements))
    k_end = int(frange[1] / fs * n_elements)

    freq = np.arange(k_low, k_end + 1) / n_elements
    t = np.arange(widx * slen, (widx + 1) * slen)

    omega = 2 * np.pi * np.outer(t, freq)

    sincol = np.sin(omega)
    coscol = np.cos(omega)

    n_freqs = k_end - k_low + 1

    d = np.zeros((slen, 2 * n_freqs))
    for idx in np.arange(n_freqs):
        d[:, 2 * idx] = coscol[:, idx]
        d[:, 2 * idx + 1] = sincol[:, idx]

    # Remove the zero column
    if k_low == 0:
        d = np.delete(d, 1, 1)

    norm = np.linalg.norm(d, 2, 0)

    d = np.divide(d, norm)

    # inner = np.matmul(d.T, d)
    # matsum = np.sum(abs(inner - np.diag(np.diag(inner))))
    # if matsum > 1e-7:
    # 	oflag = 0
    # else:
    # 	oflag = 1

    # # TODO still need to work this out analytically...
    # return d[:, 1:]
    return d
