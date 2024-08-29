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

def get_e_step_cost_func(trial_data, gamma_prev_inv, params, zs_flattened=False):
    if trial_data.ndim == 2:
        K = trial_data.shape[1]
    else:
        K = params['K']
    obs_var = params['obs_var']
    freqs = params['freqs']
    N = freqs.size
    nz_inds = params['nonzero_inds']

    def calc_obs_cost(z, data, K, N, nonzero_inds):
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
        partial_ll = b.sum()
        
        obs_cost = -partial_ll

        return obs_cost

    def calc_latent_cost(z, Gpi, K, N, nonzero_inds):
        if zs_flattened:
            z = z.reshape(-1,K)
        zs = jnp.zeros((N,K), dtype=complex)
        zs = zs.at[nonzero_inds,:].set(z)
        partial_ll = -jnp.einsum('kn,nki,ni->', zs.conj().T, Gpi, zs) # pylint: disable=invalid-unary-operand-type
        latent_cost = -partial_ll

        return latent_cost

    def cost_func(z):
        obs_cost = calc_obs_cost(z, trial_data, K, N, nz_inds) 
        latent_cost = calc_latent_cost(z, gamma_prev_inv, K, N, nz_inds)
        cost = obs_cost + latent_cost
        return cost

    return cost_func

def e_step_par(data, gamma_prev_inv, params, max_iter=5, Ups_diag=False, return_mus=False):
    
    K = data.shape[1]
    num_devices = len(jax.devices())
    Nnz = params['nonzero_inds'].size
    diag_mask = jnp.stack([jnp.eye(K) for n in range(Nnz)])

    L = data.shape[2]

    mus_outer = jnp.zeros((Nnz,K,K,L), dtype=complex)
    zs_init = jnp.zeros((Nnz,K), dtype=complex)

    def trial_optimizer(l, batch, gpi, p):

        trial_data = batch[:,:,l]
        cost_func = get_e_step_cost_func(trial_data, gpi, p)
        cost_grad = jax.grad(cost_func, holomorphic=True)
        cost_hess = jax.hessian(cost_func, holomorphic=True)
        zs_est = zs_init

        for _ in range(max_iter):
            zs_hess = cost_hess(zs_est)
            hess_sel = jnp.stack([zs_hess[n,:,n,:] for n in range(Nnz)])
            hess_sel_inv = jnp.linalg.inv(hess_sel)

            zs_grad = cost_grad(zs_est).conj()
            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_sel_inv, zs_grad)

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
            p=params)
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