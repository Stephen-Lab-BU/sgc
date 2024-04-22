import numpy as np
from numpy.fft import irfft
from cohlib.sample import gen_complex_cov, sample_complex_normal
from cohlib.utils import get_freqs, add_zero, conv_z_to_v

def sample_zs_from_Gamma(Gamma, L, seed=None):
    """Draw L samples from bcn distribution with covariances Z.
    Args:
        Z: (n_freqs, 2, 2), array of complex covs.
        L: number of samples to draw
    Returns:
        z_samples: (L, 2, n_freqs), draws from Z
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

def gen_random_mvcn_params(T, Fs, K, return_freqs=True):
    """"
    Generate (random) multivariate cc complex normal covariances.
    Args:
        T: time in seconds
        Fs: sampling frequency (in seconds; default 1000)
    Returns: 
        Gamma: (n_freqs, K, K) matrix of cc complex covariances.
    """
    # TODO handle weird times
    n = int(T / (1/Fs))
    freqs = get_freqs(T, Fs)
    n_freqs = freqs.size 
    Gamma = np.stack([gen_complex_cov(K) for _ in range(n_freqs)])

    if return_freqs:
        return Gamma, freqs

    else:
        return Gamma


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