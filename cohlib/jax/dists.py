import jax.numpy as jnp
import jax.random as jr

# TODO move LR CCN dist object to this file
# TODO move sampling functionality of CCN to dist object method

# TODO write classes for observation dist and move functionality here to methods

def sample_obs(xs, params):
    if params['obs_type'] == 'gaussian':
        sample_func = sample_obs_gaussian
    elif params['obs_type'] == 'pp_relu':
        sample_func = sample_obs_pp_relu
    elif params['obs_type'] == 'pp_log':
        sample_func = sample_obs_pp_log
    else: 
        raise NotImplementedError

    return sample_func(xs, params) 

def sample_obs_pp_poisson(xs, params, link):
    if link == 'relu':
        cif = cif_alpha_relu
    elif link == 'log':
        cif = cif_alpha_log
    else: 
        raise NotImplementedError

    seed = params['seed']
    alpha = params['alpha']
    delta = params['delta']

    ork = jr.key(seed)
    alpha = alpha
    delta = delta
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

    return obs

def sample_obs_pp_relu(xs, params): # rk, xs, alpha, C=1, delta=1e-3):
    return sample_obs_pp_poisson(xs, params, 'relu')

def sample_obs_pp_log(xs, params): # rk, xs, alpha, C=1, delta=1e-3):
    return sample_obs_pp_poisson(xs, params, 'log')

def cif_alpha_log(alphas, xs):
    return jnp.exp(alphas[None,:,None] + xs)

def cif_alpha_relu(alphas, xs):
    lams = alphas[None,:,None] + xs
    lams = lams.at[lams < 0].set(0)
    return lams


# TODO clean up params usage 
def sample_obs_gaussian(xs, params):
    ov1 = params['ov1']
    ov2 = params['ov2']
    seed = params['seed']
    print(f"Sampling Gaussian observations with variance {ov1}e^{ov2}")
    ork = jr.key(seed)
    obs_var = ov1 * 10**ov2
    params = {'obs_var': obs_var}
    obs = xs + jr.normal(ork, xs.shape)*jnp.sqrt(obs_var)

    return obs


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
# TODO add references for why this works
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
    assert jnp.all(jnp.isclose(cov.conj().T, cov, atol=1e-9))
    K = cov.shape[0]

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    # assert jnp.all(eigvals >= 0)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples

def sample_lrccn(rk, eigvecs_lr, eigvals_lr, K, L):
    R = eigvals_lr.size

    eigvecs = jnp.zeros((K,K), dtype=complex)
    eigvecs = eigvecs.at[:,:R].set(eigvecs_lr)

    eigvals = jnp.zeros(K)
    eigvals = eigvals.at[:R].set(eigvals_lr)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples


# TODO deprecate
def sample_ccn_rank1(rk, eigvec, eigval, K, L):
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
    eigvecs = jnp.zeros((K,K), dtype=complex)
    eigvecs = eigvecs.at[:,0].set(eigvec)

    eigvals = jnp.zeros(K)
    eigvals = eigvals.at[0].set(eigval)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples


# Application
# deprecated
def DEPsample_from_gamma(rk, gamma, L):
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


def naive_estimator(spikes, nonzero_inds=None):
    "spikes has shape (time, unit, trial)"
    n_f0 = jnp.fft.rfft(spikes, axis=0)
    n_f = n_f0[1:,:,:]
    naive_est = jnp.einsum('jkl,jil->jkil', n_f, n_f.conj()).mean(-1)

    if nonzero_inds is None:
        return naive_est
    else:
        return naive_est[nonzero_inds, :, :]


class LowRankCCN():
    def __init__(self, eigvals, eigvecs, dim, freqs, nonzero_inds):
        self.freqs = freqs
        self.N = freqs.size
        self.nz = nonzero_inds
        self.Nnz = nonzero_inds.size
        self.rank = eigvals.shape[1]
        self.dim = dim
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    def get_gamma(self):
        gamma = jnp.zeros((self.Nnz, self.dim, self.dim), dtype=complex)
        U_blank = jnp.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.Nnz):
            eigvals_j_lr = self.eigvals[j,:]
            eigvals_j = jnp.zeros(self.dim)
            L = jnp.diag(eigvals_j.at[:self.rank].set(eigvals_j_lr))

            eigvecs_j_lr = self.eigvecs[j,:,:]

            U = U_blank.copy()
            U = U.at[:,:self.rank].set(eigvecs_j_lr)

            gamma = gamma.at[j,:,:].set(U @ L @ U.conj().T)

        return gamma

    def get_gamma_pinv(self):
        gamma_pinv = jnp.zeros((self.Nnz, self.dim, self.dim), dtype=complex)
        U_blank = jnp.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.Nnz):
            eigvals_j_lr = 1 / self.eigvals[j,:]

            # If hard setting an eigval to 0
            eigvals_j_lr = jnp.nan_to_num(eigvals_j_lr,posinf=0,neginf=0)
            eigvals_j = jnp.zeros(self.dim, dtype=complex)
            Lam = jnp.diag(eigvals_j.at[:self.rank].set(eigvals_j_lr))

            eigvecs_j_lr = self.eigvecs[j,:,:]

            U = U_blank.copy()
            U = U.at[:,:self.rank].set(eigvecs_j_lr)

            gamma_pinv = gamma_pinv.at[j,:,:].set(U @ Lam @ U.conj().T)

        return gamma_pinv

    def sample_nz(self, rk, L):
        if self.Nnz == 1:
            samples_nz = sample_lrccn(rk, self.eigvecs[0,:,:], self.eigvals[0,:], self.dim, L)
            samples_nz = samples_nz[None,:,:]
        else:
            rksplit = jr.split(rk, self.Nnz)
            samples_nz = jnp.stack([sample_lrccn(rksplit[n], self.eigvecs[n,:,:], self.eigvals[n,:], 
                                self.dim, L) for n in range(self.Nnz)])
        return samples_nz

    def sample(self, rk, L):
        samples_nz = self.sample_nz(rk, L)
        samples = jnp.zeros((self.N,self.dim,L),dtype=complex)
        samples = samples.at[self.nz,:,:].set(samples_nz)
        return samples

