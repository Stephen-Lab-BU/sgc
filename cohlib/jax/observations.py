import os 
import multiprocessing
import jax

from functools import partial
import jax.numpy as jnp

def add_dc(x, dc):
    dc_arr = jnp.array([dc])
    with_dc = jnp.concatenate([dc_arr, x])
    return with_dc
add0 = partial(add_dc, dc=0)


def _obs_cost_gaussian(z, data, K, N, nonzero_inds, params, zs_flattened):
    obs_var = params['obs_var']
    if zs_flattened:
        z = z.reshape(-1,K)
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


def _obs_cost_pp_relu(z, data, K, N, nonzero_inds, params, zs_flattened):
    alpha = params['alpha']
    delta = params['delta']
    if zs_flattened:
        z = z.reshape(-1,K)
    zs = jnp.zeros((N,K), dtype=complex)
    zs = zs.at[nonzero_inds,:].set(z)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    if jnp.ndim(alpha) == 0:
        lams = xs + alpha
    else:
        xs + alpha[None,:,None]

    # cannot index with boolean like: lams = lams.at[lams < 0].set(jnp.nan); so:
    lams = jnp.where(lams < 0, jnp.nan, lams)

    log_lams = jnp.nan_to_num(jnp.log(lams), nan=0, neginf=0, posinf=0)
    obs_ll_calc = data*(jnp.log(delta) + log_lams) - jnp.nan_to_num(lams)*delta
    obs_ll = obs_ll_calc.sum()
    obs_cost = -obs_ll

    return obs_cost

def _obs_cost_pp_log(z, data, K, N, nonzero_inds, params, zs_flattened):
    alpha = params['alpha']
    delta = params['delta']
    if zs_flattened:
        z = z.reshape(-1,K)
    zs = jnp.zeros((N,K), dtype=complex)
    zs = zs.at[nonzero_inds,:].set(z)

    zs_0dc = jnp.apply_along_axis(add0, 0, zs)
    xs = jnp.fft.irfft(zs_0dc, axis=0)

    if jnp.ndim(alpha) == 0:
        log_lams = xs + alpha + jnp.log(delta)
    else:
        log_lams = xs + alpha[None,:,None] + jnp.log(delta)

    obs_ll_calc = (data*log_lams) - jnp.exp(log_lams)*delta
    obs_ll = obs_ll_calc.sum()
    obs_cost = -obs_ll

    return obs_cost
     


def get_e_step_cost_func(trial_data, gamma_prev_inv, params, obs_type, zs_flattened=False):
    if trial_data.ndim == 2:
        K = trial_data.shape[1]
    else:
        K = params['K']
    # obs_var = params['obs_var']
    obs_params = params['obs']
    freqs = params['freqs']
    N = freqs.size
    nz_inds = params['nonzero_inds']

    def calc_obs_cost(z, data, K, N, nonzero_inds, obs_params, zs_flattened=False):
        if obs_type == 'gaussian':
            obs_cost_func = _obs_cost_gaussian
        elif obs_type == 'pp_relu':
            obs_cost_func = _obs_cost_pp_relu
        else:
            return NotImplementedError

        obs_cost = obs_cost_func(z, data, K, N, nonzero_inds, obs_params, zs_flattened)

        return obs_cost

    def calc_latent_cost(z, Gpi, K, N, nonzero_inds):
        if zs_flattened:
            z = z.reshape(-1,K)
        zs = jnp.zeros((N,K), dtype=complex)
        zs = zs.at[nonzero_inds,:].set(z)
        latent_ll = -jnp.einsum('kn,nki,ni->', zs.conj().T, Gpi, zs) # pylint: disable=invalid-unary-operand-type
        latent_cost = -latent_ll

        return latent_cost

    def cost_func(z):
        obs_cost = calc_obs_cost(z, trial_data, K, N, nz_inds, obs_params) 
        latent_cost = calc_latent_cost(z, gamma_prev_inv, K, N, nz_inds)
        cost = obs_cost + latent_cost
        return cost

    return cost_func

def e_step_par(data, gamma_prev_inv, params, obs_type, max_iter=5, Ups_diag=False, return_mus=False, conj_hess=False):
    
    K = data.shape[1]
    num_devices = len(jax.devices())
    Nnz = params['nonzero_inds'].size
    diag_mask = jnp.stack([jnp.eye(K) for n in range(Nnz)])

    L = data.shape[2]

    mus_outer = jnp.zeros((Nnz,K,K,L), dtype=complex)
    zs_init = jnp.zeros((Nnz,K), dtype=complex)

    def trial_optimizer(trial, batch, gpi, p, obs_type):

        trial_data = batch[:,:,trial]
        cost_func = get_e_step_cost_func(trial_data, gpi, p, obs_type)
        cost_grad = jax.grad(cost_func, holomorphic=True)
        cost_hess = jax.hessian(cost_func, holomorphic=True)
        zs_est = zs_init

        for _ in range(max_iter):
            zs_hess = cost_hess(zs_est)
            hess_sel = jnp.stack([zs_hess[n,:,n,:] for n in range(Nnz)])
            if conj_hess:
                hess_sel = hess_sel.conj()
            hess_sel_inv = jnp.linalg.inv(hess_sel)

            zs_grad = cost_grad(zs_est).conj()
            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_sel_inv, zs_grad)

            # _cost = cost_func(zs_est)
            # jax.debug.breakpoint()

        zs_est = zs_est.reshape((Nnz,K))
        mu_outer = jnp.einsum('nk,ni->nki', zs_est, zs_est.conj())

        if Ups_diag is True:
            Ups = hess_sel_inv*diag_mask
        else:
            Ups = hess_sel_inv

        if return_mus:
            return [zs_est, mu_outer], Ups
        else:
            return mu_outer, Ups


    num_batches = jnp.ceil(L/num_devices).astype(int)
    mus_res = []
    Ups_res = []
    for b in range(num_batches):
        batch_data = data[:,:,b*num_devices:b*num_devices+num_devices]
        func = partial(trial_optimizer,
            batch=batch_data,
            gpi=gamma_prev_inv,
            p=params,
            obs_type=obs_type)
        if b == num_batches - 1:
            num_run_trials = b*num_devices
            remaining = L - num_run_trials
            batch_res = jax.pmap(func)(jnp.arange(remaining))
            mus_res.append(batch_res[0])
            Ups_res.append(batch_res[1])
        else:
            batch_res = jnp.arange(num_devices)
            batch_res = jax.pmap(func)(jnp.arange(num_devices))
            mus_res.append(batch_res[0])
            Ups_res.append(batch_res[1])

    Upss = jnp.moveaxis(jnp.concatenate(Ups_res, axis=0), 0, -1)
    if return_mus:
        mus_temp = [x[0] for x in mus_res]
        mus = jnp.moveaxis(jnp.concatenate(mus_temp, axis=0), 0, -1)
        mus_outer_temp = [x[1] for x in mus_res] 
        mus_outer = jnp.moveaxis(jnp.concatenate(mus_outer_temp, axis=0), 0, -1)
        return [mus, mus_outer], Upss
    else:
        mus_outer = jnp.moveaxis(jnp.concatenate(mus_res, axis=0), 0, -1)
        return mus_outer, Upss

def m_step(mus_outer, Upss):
    return (mus_outer + Upss).mean(-1)