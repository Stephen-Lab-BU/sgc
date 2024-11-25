from abc import abstractmethod, ABC
from functools import partial
from typing_extensions import Protocol

import jax
import jax.numpy as jnp

from cohlib.jax.observations import get_e_step_cost_func

class LatentFourierModel(ABC):
    """
    Abstract class to define structure all models in package follow.
    """

    @abstractmethod
    def initialize_latent(self):
        pass

    @abstractmethod
    def initialize_observations(self):
        pass

    @abstractmethod
    def fit_em(self):
        pass

    
class ToyModel(LatentFourierModel):
    def __init__(self):
        self.track = {'gamma': []}

    def initialize_latent(self, gamma, freqs, nonzero_inds):
        assert gamma.shape[1] == gamma.shape[2]
        self.gamma = gamma
        self.freqs = freqs
        self.nz = nonzero_inds
        self.K = gamma.shape[-1]

    def initialize_observations(self, obs_params, obs_type):
        self.obs_params = obs_params
        self.obs_type = obs_type

    def fit_em(self, data, num_em_iters, num_newton_iters):
        params = {'obs': self.obs_params,
                  'freqs': self.freqs,
                  'nonzero_inds': self.nz,
                  'K': self.K}


        for r in range(num_em_iters):
            print(f'EM Iter {r+1}')
            gamma_inv = jnp.zeros_like(self.gamma)
            gamma_inv_nz = jnp.linalg.inv(self.gamma[self.nz,:,:])
            gamma_inv = gamma_inv.at[self.nz,:,:].set(gamma_inv_nz)
            optimizer = JaxOptim(data, gamma_inv, params, self.obs_type, num_iters=num_newton_iters)
            mus, Upss = optimizer.run_e_step_par()

            mus_outer = jnp.einsum('nkl,nil->nkil', mus, mus.conj())

            gamma_update = jnp.zeros_like(self.gamma)
            gamma_update_nz = m_step(mus_outer, 2*Upss)
            # NOTE Upsilon is doubled - this empirically matches behavior of implementation using 'real representation'. 
            # Believe the reason is that we are effectively using only 'half' of the variables if considering our optimization 
            # in terms of CR-Calculus (Wirtinger calculus). See "The Complex Gradient Operator and the CR Calculus" (Kreutz-Delgado, 2009)
            # at https://arxiv.org/abs/0906.4835

            self.track['gamma'].append(gamma_update_nz)
            self.gamma = gamma_update.at[self.nz,:,:].set(gamma_update_nz)

        

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
        Nnz = self.Nnz
        num_devices = self.num_devices
        
        K = data.shape[1]
        L = data.shape[2]

        num_batches = jnp.ceil(L/num_devices).astype(int)
        mus_res = []
        Ups_res = []
        for b in range(num_batches):
            batch_data = data[:,:,b*num_devices:b*num_devices+num_devices]
            func = partial(self.trial_optimizer,
                batch=batch_data,
                gpi=self.gamma_inv,
                p=self.params,
                obs_type=self.obs_type)
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

        mus = jnp.moveaxis(jnp.concatenate(mus_res, axis=0), 0, -1)

        Upss = jnp.moveaxis(jnp.concatenate(Ups_res, axis=0), 0, -1)


        return mus, Upss

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

def m_step(mus_outer, Upss):
    return (mus_outer + Upss).mean(-1)



# Utility Functions - Should be moved later to alternative file
####################
class ParameterSet(Protocol):
    pass



class OptimResult():
    def __init__(self, zs_est, hess, track_zs=None, track_cost=None, track_grad=None, track_hess=None):
        self.zs_est = zs_est
        self.hess = hess
        self.track_zs = track_zs
        self.track_cost = track_cost
        self.track_grad = track_grad
        self.track_hess = track_hess

class OptimResultReal():
    def __init__(self, vs_est, hess_real):
        self.vs_est = vs_est
        self.hess_real = hess_real