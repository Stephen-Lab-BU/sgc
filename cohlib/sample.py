import math
import numpy as np
from numpy.random import multivariate_normal as mvn
import warnings

from cohlib.alg.em_sgc import transform_cov_c2r


def sample_spikes_from_xs(lams, C, delta=1, group_axis=1, obs_model='bernoulli'):
    if obs_model == 'bernoulli':
        sampler = _c_sample_func_bernoulli(C)
    elif obs_model == 'poisson':
        sampler = _c_sample_func_poisson(C)
    else:
        raise ValueError
    samples = np.apply_along_axis(sampler, group_axis, lams*delta)
    return samples

def _c_sample_func_poisson(C):
    def func(x):
        reps = np.tile(x, C).reshape(C,-1)
        samples = np.random.poisson(reps)
        return samples
    return func

def _c_sample_func_bernoulli(C):
    def func(x):
        reps = np.tile(x, C).reshape(C,-1)
        samples = np.random.binomial(1, reps)
        return samples
    return func

def sample_complex_normal(cov, n, seed=None):
    rcov = transform_cov_c2r(cov)
    rdim = rcov.shape[0]
    rhalfdim = int(rdim/2)
    if seed is not None:
        if np.isscalar(seed):
            np.random.seed(seed)
            zr_samps = _mvn_sample_ignore_warning(rdim, rcov, n)
        else:
            rng = np.random.default_rng(seed)
            zr_samps = _mvn_sample_ignore_warning(rdim, rcov, n, rng)
    else:
        zr_samps = _mvn_sample_ignore_warning(rdim, rcov, n)

    zc_samples = zr_samps[:,:rhalfdim] + zr_samps[:,rhalfdim:]*1j

    return zc_samples.swapaxes(0,1)

def _mvn_sample_ignore_warning(rdim, rcov, n, rng=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if rng is None:
            zr_samps = mvn(np.zeros(rdim), rcov, n)
        else:
            zr_samps = rng.multivariate_normal(np.zeros(rdim), rcov, n)

    return zr_samps

def sample_complex_normal_dep(cov, n):
    m = cov.shape[0]
    L = np.linalg.cholesky(cov)
    temp = np.random.randn(m,n) + 1j*np.random.randn(m,n)
    
    sample = L @ temp
    
    return sample

def gen_complex_cov(K):
    A = np.random.randn(K,K) + 1j*np.random.randn(K,K)
    R = np.conj(A).T @ A
    
    return R

def sig_from_complex(c, time, freq):
    r = np.abs(c)
    theta = np.angle(c)
    # sig = r*np.exp(1j*2*np.pi*freq*theta*time)
    
    cos = np.array([math.cos(2*np.pi*freq*t + theta) for t in time])
    sin = np.array([math.sin(2*np.pi*freq*t + theta) for t in time])
    
    return r*(cos + 1j*sin)
    
def sig_from_real(a, b, time, freq):
    cos = np.array([a*math.cos(2*np.pi*freq*t) for t in time])
    sin = np.array([b*math.sin(2*np.pi*freq*t) for t in time])
    sig = cos + sin
    return 1/2*sig