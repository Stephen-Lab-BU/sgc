import numpy as np
from numpy.fft import irfft
from cohlib.sample import gen_complex_cov, sample_complex_normal
from cohlib.utils import get_freqs, add_zero

# test: delete
# TODO make this a class!
def gen_bcn_params(T, Fs=1000, return_freqs=True):
    """"
    Generate (random) bivariate cc complex normal covariances.
    Args:
        T: time in seconds
        Fs: sampling frequency (in seconds; default 1000)
    Returns: 
        Z: (n_freqs, 2, 2) matrix of cc complex covariances.
    """
    # TODO handle weird times
    n = int(T / (1/Fs))
    freqs = get_freqs(T, Fs)
    n_freqs = freqs.size 
    d = 2
    Z = np.stack([gen_complex_cov(d) for _ in range(n_freqs)])

    if return_freqs:
        return Z, freqs

    else:
        return Z

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

# TODO rename - this is sampling
def gen_bcn_time_obs(Z, L, return_zs=False, norm='ortho'):
    z_samples = sample_from_Z(Z, L)
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