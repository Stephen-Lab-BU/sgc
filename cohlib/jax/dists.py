import jax.numpy as jnp
import jax.random as jr


def sample_obs(ocfg, xs):
    if ocfg.obs_type == 'gaussian':
        sample_func = sample_obs_gaussian
    elif ocfg.obs_type == 'pp_relu':
        sample_func = sample_obs_pp_relu
    elif ocfg.obs_type == 'pp_log':
        sample_func = sample_obs_pp_log
    else: 
        raise NotImplementedError

    obs, params = sample_func(ocfg, xs)
    return obs, params

def sample_obs_pp_poisson(link, ocfg, xs):
    if link == 'relu':
        cif = cif_alpha_relu
    elif link == 'log':
        cif = cif_alpha_log
    else: 
        raise NotImplementedError

    ork = jr.key(ocfg.seed)
    alpha = ocfg.alpha
    delta = ocfg.delta
    C = 1
    K = xs.shape[1]
    if jnp.ndim(alpha) == 0:
        alphas = jnp.ones(K)*alpha
    else:
        assert alpha.ndim == 1
        assert alpha.size == K
        alphas = alpha
    lams_single = cif(alphas, xs)
    lams = jnp.stack([lams_single for _ in range(C)], axis=1)
    samples = jr.poisson(ork, lams*delta)
    obs = samples.squeeze()
    params = {'alpha': alpha, 'delta': delta}

    return obs, params

def sample_obs_pp_relu(ocfg, xs): # rk, xs, alpha, C=1, delta=1e-3):
    return sample_obs_pp_poisson('relu', ocfg, xs)

def sample_obs_pp_log(ocfg, xs): # rk, xs, alpha, C=1, delta=1e-3):
    return sample_obs_pp_poisson('log', ocfg, xs)

def cif_alpha_log(alphas, xs):
    return jnp.exp(alphas[None,:,None] + xs)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams = lams.at[lams < 0].set(0)
    return lams


def sample_obs_gaussian(ocfg, xs):
    print(f"Sampling Gaussian observations with variance {ocfg.ov1}e^{ocfg.ov2}")
    ork = jr.key(ocfg.seed)
    obs_var = ocfg.ov1 * 10**ocfg.ov2
    params = {'obs_var': obs_var}
    obs = xs + jr.normal(ork, xs.shape)*jnp.sqrt(obs_var)

    return obs, params


def sample_spikes_from_lams(rk, lams, C, delta=1, group_axis=1, obs_model='bernoulli'):
    if obs_model == 'bernoulli':
        sampler = _c_sample_func_bernoulli(rk, C)
    elif obs_model == 'poisson':
        sampler = _c_sample_func_poisson(rk, C)
    else:
        raise ValueError
    samples = jnp.apply_along_axis(sampler, group_axis, lams*delta)
    return samples

def _c_sample_func_poisson(rk, C):
    def func(x):
        reps = jnp.tile(x, C).reshape(C,-1)
        samples = jr.poisson(rk, reps)
        return samples
    return func

def _c_sample_func_bernoulli(rk, C):
    def func(x):
        reps = jnp.tile(x, C).reshape(C,-1)
        samples = jr.binomial(rk, 1, reps)
        return samples
    return func


# Latent
# TODO 
# generalize
# - create generic 'ccn_sample' function and use

# Generics
def sample_ccn(rk, cov, L):
    """
    Generate L samples from K-dimensional multivariate complex 
    normal (circular symmetric). 'cov' must be psd Hermitian.

    Args:
        rk: jax random key
        cov: (K,K) array PSD Hermitian covariance matrix
        L: number of samples

    Returns:
        samples: (K,L) array (complex)
    
    """
    assert jnp.all(cov.conj().T == cov)
    K = cov.shape[0]

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    assert jnp.all(eigvals >= 0)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples

# Application
def sample_from_gamma(rk, gamma, L):
    """
    Generate L samples from multivariate complex normal (circular symmetric). 
    Gamma is assumed to be indepedent across frequencies (block-diagonal).

    Args:
        gamma: (N,K,K) array where N is frequencies in nyquist 
            range, K is number of units
        L: number of samples to generate
        rk: jax random key
    Return:
        samples: (N,K) array (complex)
    """
    N = gamma.shape[0]

    rksplit = jr.split(rk,N)
    samples = jnp.stack([sample_ccn(rksplit[n], gamma[n,:,:], L) for n in range(N)])
    # samples = jnp.stack([sample_ccn(rk+n, gamma[n,:,:], L) for n in range(N)])

    return samples