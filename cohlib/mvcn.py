import numpy as np
from numpy.fft import irfft, rfftfreq
from cohlib.sample import gen_complex_cov, sample_complex_normal
from cohlib.utils import conv_z_to_v

def sample_zs_from_Gamma(Gamma, L, seed=None):
    """Draw L samples from K-dim distribution with covariances Z.
    Args:
        Z: (n_freqs, K, K), array of complex covs.
        L: number of samples to draw
    Returns:
        z_samples: (L, K, n_freqs), draws from Z
    """
    n_freqs = Gamma.shape[0]
    if seed is not None:
        if np.isscalar(seed):
            np.random.seed(1)
            z_list = [sample_complex_normal(Gamma[j,:,:], L) for j in range(n_freqs)]
        else:
            raise NotImplementedError
    else:
        z_list = [sample_complex_normal(Gamma[j,:,:], L) for j in range(n_freqs)]
    z_join = np.stack(z_list)
    z_draws = np.swapaxes(z_join, 0, 2)

    return z_draws

def gen_random_mvcn_params(T, Fs, K, cut_freq=None, return_freqs=True):
    """"
    Generate (random) multivariate cc complex normal covariances.
    Args:
        T: time in seconds
        Fs: sampling frequency (in seconds; default 1000)
    Returns: 
        Gamma: (n_freqs, K, K) matrix of cc complex covariances.
    """
    # TODO handle weird times
    freqs = get_freqs(T, Fs)
    if cut_freq is not None:
        cut_freq_ind = np.where(freqs > cut_freq)[0][0]
        freqs = freqs[:cut_freq_ind]

    n_freqs = freqs.size 
    Gamma = np.stack([gen_complex_cov(K) for _ in range(n_freqs)])

    if return_freqs:
        return Gamma, freqs

    else:
        return Gamma


# TODO deprecated - remove
def sample_from_Z(Z, L):
    """Draw L samples from bcn distribution with covariances Z.
    Args:
        Z: (n_freqs, 2, 2), array of complex covs.
        L: number of samples to draw
    Returns:
        z_samples: (L, 2, n_freqs), draws from Z
    """
    n_freqs = Z.shape[0]
    z_list = [sample_complex_normal(Z[j,:,:], L) for j in range(n_freqs)]
    z_join = np.stack(z_list)
    z_draws = np.swapaxes(z_join, 0, 2)

    return z_draws

def sample_mvcn_time_obs(Gamma, L, freqs, Wv, dc, return_all=True, support_filt=None):
    K = Gamma.shape[1]
    J = freqs.size
    zs = np.zeros((L,K,J+1),dtype=complex)
    # dc_rand = 5*np.random.randn(L,K)
    # zs[:,:,0] = dc[None,:] + dc_rand
    zs[:,:,0] = dc[None,:] 
    if Gamma.shape[0] != J:
        num_freqs_Gamma = Gamma.shape[0]
        band_samples = sample_zs_from_Gamma(Gamma, L)
        support_filt_dc = np.zeros(J+1).astype(bool)

        if support_filt is None:
            support_filt_dc[1:num_freqs_Gamma+1] = True
        else:
            support_filt_dc[1:] = support_filt

        zs[:,:,support_filt_dc] = band_samples

    else:
        samples = sample_zs_from_Gamma(Gamma, L)
        zs[:,:,1:] = samples

    vs = conv_z_to_v(zs, axis=2, dc=True)

    xs = np.einsum('ij,abj->abi', Wv, vs)

    if return_all:
        return xs, vs, zs
    else:
        return xs

def sample_mvcn_time_obs_nodc(Gamma, L, freqs, Wv, return_all=True, support_filt=None):
    K = Gamma.shape[1]
    J = freqs.size
    zs = np.zeros((L,K,J),dtype=complex)
    # dc_rand = 5*np.random.randn(L,K)
    # zs[:,:,0] = dc[None,:] + dc_rand
    if Gamma.shape[0] != J:
        num_freqs_Gamma = Gamma.shape[0]
        band_samples = sample_zs_from_Gamma(Gamma, L)
        support_filt = np.zeros(J).astype(bool)

        if support_filt is None:
            support_filt[:num_freqs_Gamma] = True

        zs[:,:,support_filt] = band_samples

    else:
        samples = sample_zs_from_Gamma(Gamma, L)
        zs = samples

    vs = conv_z_to_v(zs, axis=2, dc=False)

    xs = np.einsum('ij,abj->abi', Wv, vs)

    if return_all:
        return xs, vs, zs
    else:
        return xs

# NOTE for backwards compatability
def gen_bcn_params(T, Fs=1000, K=2, return_freqs=True):
    return gen_random_mvcn_params(T, Fs, K, return_freqs=True)
def sample_bcn_time_obs(Gamma, L, return_zs=False, norm='ortho'):
    z_samples = sample_zs_from_Gamma(Gamma, L)
    x_F = z_samples[:,0,:]
    y_F = z_samples[:,1,:]
    x_F0 = add_zero(x_F)
    y_F0 = add_zero(y_F)
    x = irfft(x_F0, axis=1, norm=norm)
    y = irfft(y_F0, axis=1, norm=norm)

    if return_zs:
        return x, y, z_samples
    else:
        return x, y


# TODO test
def get_freqs(window_size, Fs):
    """
    Get frequency values corresponding to Fourier transform.
    Args:
        window_size (float): length of window in *seconds*
        Fs (int): sampling rate
    """
    n = int(window_size / (1 / Fs))
    freqs = rfftfreq(n, d=1 / Fs)
    return freqs[1:]

def add_zero(x_F, axis=1):
    zero = np.array([0 + 1j * 0])
    n_trials = x_F.shape[0]
    zeros = np.repeat(zero, n_trials)
    new_x_F = np.concatenate([zeros[:, None], x_F], axis=axis)
    return new_x_F


def estimate_coherence(xf,yf, mag_sq=True):
    """
    Estimate coherence for a single frequency range from observed complex coefs. 
    Args:
        xf: (n_trials,) array of complex coefficients signal 1
        yf: (n_trials,) array of complex coefficients signal 2
        mag_sq: (bool) optional - return mean-squared coherence 
    Returns:
        coh: coherence estimate
    """

    Sxy = xf * yf.conj()
    Sxx = xf * xf.conj()
    Syy = yf * yf.conj()

    if mag_sq:
        num = np.abs(Sxy.mean(0))**2
        denom = Sxx.mean(0).real * Syy.mean(0).real

    else:
        num = np.abs(Sxy.mean(0))
        a = np.sqrt(Sxx.mean(0).real)
        b = np.sqrt(Syy.mean(0).real)
        denom = a*b

    coh  = num/denom

    return coh

def thr_coherence(Gamma, mag_sq=True):
    """
    Calculate theoretical coherence from covariance matrices. 
    Args:
        Gamma: (n_freqs, 2, 2) array of complex bcn covariance matrices
    Returns:
        t_coh: (n_freqs,) array of coherence values
    """

    if mag_sq:
        num = np.abs(Gamma[:,0,1])**2
        a = np.abs(Gamma[:,0,0])
        b = np.abs(Gamma[:,1,1])

    else:
        num = np.abs(Gamma[:,0,1])
        a = np.abs(Gamma[:,0,0])
        b = np.abs(Gamma[:,1,1])
        a, b = np.sqrt(a), np.sqrt(b)

    denom = a*b
    t_coh = num/denom

    return t_coh