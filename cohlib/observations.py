import jax
import jax.random as jr
import jax.numpy as jnp

from cohlib.utils import add0

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
        cif = cif_mu_relu
    elif link == 'log':
        cif = cif_mu_log
    else: 
        raise NotImplementedError

    seed = params['seed']
    mu = params['mu']
    delta = params['delta']

    ork = jr.key(seed)
    mu = mu
    delta = delta
    C = 1
    K = xs.shape[1]
    if jnp.ndim(mu) == 0:
        mus = jnp.ones(K)*mu
    else:
        assert mu.ndim == 1
        assert mu.size == K
        mus = mu
    lams_single = cif(mus, xs)
    lams = jnp.stack([lams_single for _ in range(C)], axis=1)
    samples = jr.poisson(ork, lams*delta)
    obs = samples.squeeze()
    params = {'mu': mu, 'delta': delta}

    return obs

def sample_obs_pp_relu(xs, params): # rk, xs, mu, C=1, delta=1e-3):
    return sample_obs_pp_poisson(xs, params, 'relu')

def sample_obs_pp_log(xs, params): # rk, xs, mu, C=1, delta=1e-3):
    return sample_obs_pp_poisson(xs, params, 'log')

def cif_mu_log(mus, xs):
    return jnp.exp(mus[None,:,None] + xs)

def cif_mu_relu(mus, xs):
    lams = mus[None,:,None] + xs
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



def _obs_cost_gaussian(z, data, K, N, nonzero_inds, params):
    obs_var = params['obs_var']
    zs = jnp.zeros((N,K), dtype=complex)
    zs = zs.at[nonzero_inds,:].set(z)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)
    err = (data - xs)

    # partial_ll = -0.5 * (err**2 * 1/obs_var).sum()
    a = -0.5 * (err**2 * 1/obs_var)
    b = a.sum(0)
    obs_ll = b.sum()
    
    obs_cost = -obs_ll

    return obs_cost

# TODO deprecate if unused
@jax.jit
def jitted_pp_relu_calc_cost(xs, mu, data, delta):
    lams = xs + mu

    # if jnp.ndim(mu) == 0:
    #     lams = xs + mu
    # else:
    #     lams = xs + mu[None,:,None]

    # cannot index with boolean like: lams = lams.at[lams < 0].set(jnp.nan); so:
    lams = jnp.where(lams < 0, jnp.nan, lams)

    log_lams = jnp.nan_to_num(jnp.log(lams), nan=0, neginf=0, posinf=0)
    obs_ll_calc = data*(jnp.log(delta) + log_lams) - jnp.nan_to_num(lams)*delta
    obs_ll = obs_ll_calc.sum()
    obs_cost = -obs_ll

    return obs_cost

def _obs_cost_pp_relu(z, data, K, N, nonzero_inds, params):
    mu = params['mu']
    delta = params['delta']
    zs = jnp.zeros((N,K), dtype=complex)
    zs = zs.at[nonzero_inds,:].set(z)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    # obs_cost = jitted_pp_relu_calc_cost(xs, mu, data, delta)
    lams = xs + mu

    # cannot index with boolean like: lams = lams.at[lams < 0].set(jnp.nan); so:
    lams = jnp.where(lams < 0, jnp.nan, lams)

    log_lams = jnp.nan_to_num(jnp.log(lams), nan=0, neginf=0, posinf=0)
    obs_ll_calc = data*(jnp.log(delta) + log_lams) - jnp.nan_to_num(lams)*delta
    obs_ll = obs_ll_calc.sum()
    obs_cost = -obs_ll

    return obs_cost

# def _obs_cost_pp_log(z, data, K, N, nonzero_inds, params):
#     mu = params['mu']
#     delta = params['delta']
#     zs = jnp.zeros((N,K), dtype=complex)
#     zs = zs.at[nonzero_inds,:].set(z)

#     zs_0dc = jnp.apply_along_axis(add0, 0, zs)
#     xs = jnp.fft.irfft(zs_0dc, axis=0)

#     log_lams = xs + mu
#     lams = jnp.exp(log_lams)

#     obs_ll_calc = data*(jnp.log(delta) + log_lams) - lams*delta
#     obs_ll = obs_ll_calc.sum()
#     obs_cost = -obs_ll

#     # log_lams = xs + mu + jnp.log(delta)

#     # obs_ll_calc = (data*log_lams) - jnp.exp(log_lams)*delta
#     # obs_ll = obs_ll_calc.sum()
#     # obs_cost = -obs_ll

#     return obs_cost

def _obs_cost_pp_log(z, data, K, N, nonzero_inds, params):
    mu = params['mu']
    delta = params['delta']
    zs = jnp.zeros((N,K), dtype=complex)
    zs = zs.at[nonzero_inds,:].set(z)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    log_lams = xs + mu[None,:]
    lams = jnp.exp(log_lams)

    obs_ll_calc = data*(jnp.log(delta) + log_lams) - lams*delta
    obs_ll = obs_ll_calc.sum()
    obs_cost = -obs_ll

    # log_lams = xs + mu + jnp.log(delta)

    # obs_ll_calc = (data*log_lams) - jnp.exp(log_lams)*delta
    # obs_ll = obs_ll_calc.sum()
    # obs_cost = -obs_ll

    return obs_cost
     

