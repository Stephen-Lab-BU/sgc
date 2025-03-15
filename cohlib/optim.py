from functools import partial

import jax
import jax.numpy as jnp

from cohlib.jax.observations import _obs_cost_gaussian, _obs_cost_pp_log, _obs_cost_pp_relu

# deprecate
# TODO Rename to 'Laplace approx' and clarify 
class JaxOptim():
    def __init__(self, data, gamma_inv, params, obs_type, track=False, num_iters=10):
        self.data = data
        self.gamma_inv = gamma_inv
        self.K = gamma_inv.shape[-1]
        self.params = params
        self.nz = params['nonzero_inds']
        self.Nnz = self.nz.size
        self.track = track
        self.Ups_diag = False
        self.num_iters = num_iters
        self.obs_type = obs_type
        self.num_devices = len(jax.devices())

        # TODO add Note on holomorphic
        self.cost_func = get_e_step_cost_func(data, gamma_inv, params, obs_type)
        self.cost_grad = jax.grad(self.cost_func, holomorphic=True)
        self.cost_hess = jax.hessian(self.cost_func, holomorphic=True)        

    def eval_cost(self, zs):
        cost = self.cost_func(zs)
        grad = self.cost_grad(zs)
        hess = self.cost_hess(zs)
        hess_sel = jnp.stack([hess[n,:,n,:] for n in range(self.Nnz)])

        return cost, grad.conj(), hess_sel


    def run_e_step_par(self):
        data = self.data
        num_devices = self.num_devices
        L = data.shape[2]

        num_batches = jnp.ceil(L/num_devices).astype(int)
        alphas_res = []
        Ups_res = []
        # print(f'Num Devices: {num_devices}')
        # print(f'L: {L}, num_batches: {num_batches}')
        for b in range(num_batches):
            batch_data = data[:,:,b*num_devices:b*num_devices+num_devices]
            print(f'Running batch {b}: trials {b*num_devices + 1 } - {b*num_devices+batch_data.shape[2] }')
            func = partial(self.trial_optimizer,
                batch=batch_data,
                gpi=self.gamma_inv,
                p=self.params,
                obs_type=self.obs_type)
            if b == num_batches - 1:
                num_run_trials = b*num_devices
                remaining = L - num_run_trials
                batch_res = jax.pmap(func)(jnp.arange(remaining))
                alphas_res.append(batch_res[0])
                Ups_res.append(batch_res[1])
            else:
                batch_res = jnp.arange(num_devices)
                batch_res = jax.pmap(func)(jnp.arange(num_devices))
                alphas_res.append(batch_res[0])
                Ups_res.append(batch_res[1])

        # print(f'length output = {len(alphas_res)}')
        # alphas_temp = jnp.concatenate(alphas_res, axis=0)
        alphas = jnp.moveaxis(jnp.concatenate(alphas_res, axis=0), 0, -1)

        # print(f'length output = {len(Ups_res)}')
        # Upss_temp = jnp.concatenate(Ups_res, axis=0) 
        # print(f'Upss cat shape: {Upss_temp.shape}')
        Upss = jnp.moveaxis(jnp.concatenate(Ups_res, axis=0), 0, -1)
        # print(f'Upss dim reordered shape: {Upss.shape}')


        return alphas, Upss

    # TODO review and remove if no longer needed
    def run_e_step_par_ts(self, data, num_devices):
        L = data.shape[2]

        num_batches = jnp.ceil(L/num_devices).astype(int)
        print(f'num_batches = {num_batches}')
        alphas_res = []
        Ups_res = []
        for b in range(num_batches):
            batch_data = data[:,:,b*num_devices:b*num_devices+num_devices]
            print(f'batch shape: {batch_data.shape}')
            func = partial(self.trial_optimizer,
                batch=batch_data,
                gpi=self.gamma_inv,
                p=self.params,
                obs_type=self.obs_type)
            if b == num_batches - 1:
                num_run_trials = b*num_devices
                remaining = L - num_run_trials
                batch_res = jax.pmap(func)(jnp.arange(remaining))
                alphas_res.append(batch_res[0])
                Ups_res.append(batch_res[1])
            else:
                batch_res = jnp.arange(num_devices)
                batch_res = jax.pmap(func)(jnp.arange(num_devices))
                alphas_res.append(batch_res[0])
                Ups_res.append(batch_res[1])

        return alphas_res, Ups_res

    def trial_optimizer(self, trial, batch, gpi, p, obs_type):
        """
        # TODO finish note
        NOTE Cost function is C^n->R, by definition not holomorphic. 
        We can take a gradient, but to take Hessian, 
        we are effectively taking Jacobian of C^n->C^n function. 
        Jax only allows us to do this we declare that cost_func is holomorphic.
    
        See references: 
        https://github.com/jax-ml/jax/discussions/10155
        https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
        """

        trial_data = batch[:,:,trial]
        cost_func = get_e_step_cost_func(trial_data, gpi, p, obs_type)
        cost_grad = jax.jit(jax.grad(cost_func, holomorphic=True))
        cost_hess = jax.jit(jax.hessian(cost_func, holomorphic=True))
        Nnz = self.Nnz
        K = self.K

        zs_init = jnp.zeros((Nnz,K), dtype=complex)
        zs_est = zs_init
        for _ in range(self.num_iters):
            zs_hess = cost_hess(zs_est)
            zs_grad = cost_grad(zs_est).conj()
            hess_sel = jnp.stack([zs_hess[n,:,n,:] for n in range(Nnz)]).conj()
            zs_est, hess_sel_inv = newton_step(zs_est, zs_grad, hess_sel)

            # zs_hess = cost_hess(zs_est)
            # # TODO finish note
            # # NOTE Jax gives us a gradient and Hessian - we need to take the
            # # conjugate of both when using Newton's method
            # hess_sel = jnp.stack([zs_hess[n,:,n,:] for n in range(Nnz)]).conj()
            # hess_sel_inv = jnp.linalg.inv(hess_sel)

            # zs_grad = cost_grad(zs_est).conj()
            # zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_sel_inv, zs_grad)

        mu = zs_est

        # TODO Remove after testing 
        if self.Ups_diag is True:
            diag_mask = jnp.stack([jnp.eye(K) for n in range(Nnz)])
            Ups = hess_sel_inv*diag_mask
        else:
            Ups = hess_sel_inv

        return mu, Ups

@jax.jit
def newton_step(zs_est, zs_grad, hess_sel):
    # TODO finish note
    # NOTE Jax gives us a gradient and Hessian - we need to take the
    # conjugate of both when using Newton's method
    hess_sel_inv = jnp.linalg.inv(hess_sel)

    zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_sel_inv, zs_grad)

    return zs_est, hess_sel_inv 

# TODO this works for now, but next step should improve latent/observation spec 
# TODO make calculating obs_cost and latent_cost methods of CCN and Obs objects
# TODO make Obs for convenience, be able to access *trial* data within Obs object easy
    # or even make Obs object a collection of trials 
def get_e_step_cost_func(trial_data, gamma_prev_inv, params, obs_type):
    if trial_data.ndim == 2:
        K = trial_data.shape[1]
    else:
        K = params['K']
    # obs_var = params['obs_var']
    obs_params = params['obs']
    freqs = params['freqs']
    N = freqs.size
    nz_inds = params['nonzero_inds']

    def calc_obs_cost(z, data, K, N, nonzero_inds, obs_params):
        if obs_type == 'gaussian':
            obs_cost_func = _obs_cost_gaussian
        elif obs_type == 'pp_relu':
            obs_cost_func = _obs_cost_pp_relu
        elif obs_type == 'pp_log':
            obs_cost_func = _obs_cost_pp_log
        else:
            return NotImplementedError

        obs_cost = obs_cost_func(z, data, K, N, nonzero_inds, obs_params)

        return obs_cost

    def calc_latent_cost(z, Gpi, K, N, nonzero_inds):
        zs = jnp.zeros((N,K), dtype=complex)
        zs = zs.at[nonzero_inds,:].set(z)
        latent_ll = -jnp.einsum('kn,nki,ni->', zs.conj().T, Gpi, zs) # pylint: disable=invalid-unary-operand-type
        latent_cost = -latent_ll

        return latent_cost

    @jax.jit
    def cost_func(z):
        obs_cost = calc_obs_cost(z, trial_data, K, N, nz_inds, obs_params) 
        latent_cost = calc_latent_cost(z, gamma_prev_inv, K, N, nz_inds)
        cost = obs_cost + latent_cost
        return cost

    return cost_func