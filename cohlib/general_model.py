from abc import abstractmethod, ABC
from functools import partial

import jax
import jax.random as jr
import jax.numpy as jnp

from cohlib.optim import JaxOptim
from cohlib.latent import LowRankCCN, CCN

# TODO make this actually useful or discard
class LatentFourierModel(ABC):
    """
    Abstract class to define structure all models in package follow.
    """

    @abstractmethod
    def initialize_latent(self):
        pass
        # self.latent = create_latent(dist_type, params)

    @abstractmethod
    def initialize_observations(self):
        pass

    @abstractmethod
    def fit_em(self):
        pass

class GeneralToyModel(LatentFourierModel):
    def __init__(self, track_params=True):
        self.track = {'ccn': []}
        self.track_ccn = track_params

    def initialize_latent(self, ccn):
        self.ccn = ccn
        self.Kr = self.ccn.rank
        self.K = self.ccn.dim
        self.J = ccn.Nnz
        self.freqs = ccn.freqs
        self.nz = ccn.nz
        self.Nnz = self.freqs.size


    def initialize_observations(self, obs_params, obs_type):
        self.obs_params = obs_params
        self.obs_type = obs_type


    def fit_em(self, data, fit_params):
        
        num_em_iters = fit_params['num_em_iters']
        num_newton_iters = fit_params['num_newton_iters']
        m_step_option = fit_params['m_step_option']
        m_step_params = fit_params.get('m_step_params', {})
        fixed_params = fit_params.get('fixed_params', {})

        m_step_params['fixed_params'] = fixed_params


        params = {'obs': self.obs_params,
                  'freqs': self.freqs,
                  'nonzero_inds': self.nz,
                  'K': self.K}

        if m_step_option == 'low-rank-eigh':
            self.m_step = m_step_lowrank_eigh
            self.m_step_params = m_step_params
        elif m_step_option == 'full-rank-standard':
            self.m_step = m_step_fullrank
            self.m_step_params = None
        else:
            raise NotImplementedError
    
        if self.track_ccn is True:
            self.track['ccn'].append(self.ccn)

        for r in range(num_em_iters):
            self.r = r
            print(f'EM Iter {r+1}')
            gamma_inv = jnp.zeros((self.Nnz, self.K, self.K), dtype=complex)
            if self.ccn.rank == self.ccn.dim:
                gamma_inv_nz = self.ccn.get_gamma_inv()
            else:
                gamma_inv_nz = self.ccn.get_gamma_pinv()
            gamma_inv = gamma_inv.at[self.nz,:,:].set(gamma_inv_nz)
            optimizer = JaxOptim(data, gamma_inv, params, self.obs_type, num_iters=num_newton_iters)
            alphas, Upss = optimizer.run_e_step_par()
            self.alphas = alphas
            self.Upss = Upss

            alphas_outer = jnp.einsum('nkl,nil->nkil', alphas, alphas.conj())


            if self.ccn.rank == self.ccn.dim:
                gamma_update = self.m_step(alphas_outer, 2*Upss, self.m_step_params)
                inv_flag = self.ccn.inv_flag

                ccn_update = CCN(gamma_update, freqs=self.freqs, nonzero_inds=self.nz, inv_flag=inv_flag)

            else:
                m_step_params['ccn_prev'] = self.ccn
                eigvals_update, eigvecs_update = self.m_step(alphas_outer, 2*Upss, self.m_step_params)

                # TODO review / finalize this
                # NOTE Upsilon is doubled - this empirically matches behavior of implementation using 'real representation'. 
                # Believe the reason is that we are effectively using only 'half' of the variables if considering our optimization 
                # in terms of CR-Calculus (Wirtinger calculus). See "The Complex Gradient Operator and the CR Calculus" (Kreutz-Delgado, 2009)
                # at https://arxiv.org/abs/0906.4835

                # eigvecs_update = rotate_eigvecs(eigvecs_update)
                # eigvecs_update = stdize_eigvecs(eigvecs_update)
                print('eigvec update:')
                print(jnp.round(eigvecs_update,2))
                print('eigval update:')
                print(jnp.round(eigvals_update,2))

                ccn_update = LowRankCCN(eigvals_update, eigvecs_update, dim=self.K, freqs=self.freqs, nonzero_inds=self.nz)

            if self.track_ccn is True:
                self.track['ccn'].append(ccn_update)
            self.ccn = ccn_update

def m_step_fullrank(alphas_outer, Upss, options=None):
    return (alphas_outer + Upss).mean(-1)

def m_step_lowrank_eigh(alphas_outer, Upss, params):
    lrccn_prev = params['ccn_prev']
    rank = lrccn_prev.rank
    J = lrccn_prev.Nnz
    fixed_params = params['fixed_params']

    Sigma_ests = (alphas_outer + Upss)
    eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
    eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)

    for j in range(J):
        gamma_ests_j = Sigma_ests[j,:,:,:].mean(-1)
        eigvals_eigh_j, eigvecs_eigh_j = jnp.linalg.eigh(gamma_ests_j)

        if 'eigvals' in fixed_params.keys():
            eigvals_update = fixed_params['eigvals']
        else:
            eigvals_update = eigvals_update.at[j,:rank].set(eigvals_eigh_j[-rank:][::-1])

        if 'eigvecs' in fixed_params.keys():
            eigvecs_update = fixed_params['eigvecs']
        else:
            eigvecs_update = eigvecs_update.at[j,:,:rank].set(eigvecs_eigh_j[:,-rank:][:,::-1])

        return eigvals_update, eigvecs_update

