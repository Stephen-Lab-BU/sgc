import os
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr
from cohlib.jax.dists import sample_from_gamma

from cohlib.utils import gamma_root, pickle_open
from cohlib.conv import conv_v_to_z, conv_z_to_v
from cohlib.alg.em_sgc import construct_Gamma_full_real, deconstruct_Gamma_full_real
from cohlib.jax.gaussian_obs import get_e_step_cost_func
from cohlib.alg.laplace_gaussian_obs import TrialDataGaussian, GaussianTrial, GaussianTrialMod

from pathlib import Path
from omegaconf import OmegaConf

def load_results(paths, ovs_sel, **kwargs):
    Lpaths = get_model_paths(paths, **kwargs)
    results = {}
    obs_vars = []
    for path in Lpaths:
        cfg_path = os.path.join(path, '.hydra/config.yaml')
        cfg = OmegaConf.load(cfg_path)
        obs_var = cfg.obs.ov2
        if ovs_sel is not None:
            if obs_var in ovs_sel:
                obs_vars.append(obs_var)

    ovs = jnp.array(obs_vars)
    if ovs.size == jnp.unique(ovs).size:

        for path in Lpaths:
            cfg_path = os.path.join(path, '.hydra/config.yaml')
            cfg = OmegaConf.load(cfg_path)
            obs_var = cfg.obs.ov2

            res = pickle_open(os.path.join(path, 'res.pickle'))
            res['cfg'] = cfg
            results[obs_var] = res

    else:
        print('Duplicates found for obs_var - returning empty dict.')

    return results

def get_model_paths(paths, **kwargs):
    sel_paths = []
    for path in paths:
        _dir = Path(path)
        for i, exp in enumerate(_dir.glob('*')):
            cfg_path = os.path.join(exp, '.hydra/config.yaml')
            cfg = OmegaConf.load(cfg_path)

            if cfg.latent.L == kwargs['L']:
                if cfg.model.emiters == kwargs['emiters']:
                    if cfg.model.init == kwargs['init']:
                        if cfg.model.scale_init == kwargs['scale_init']:
                            if kwargs['supp'] is not None:
                                if 'support' in list(cfg.model.keys()):
                                    if cfg.model.support == kwargs['supp']:
                                            sel_paths.append(exp)
                            else:
                                if 'support' not in list(cfg.model.keys()):
                                    if cfg.model.emiters == kwargs['emiters']:
                                        sel_paths.append(exp)
                                else:
                                    if cfg.model.support == kwargs['supp']:
                                            sel_paths.append(exp)
    return sel_paths

class OptimResult():
    def __init__(self, zs_est, hess, track_zs=None, track_cost=None, track_grad=None, track_hess=None):
        self.zs_est = zs_est
        self.hess = hess
        self.track_zs = track_zs
        self.track_cost = track_cost
        self.track_grad = track_grad
        self.track_hess = track_hess

class JaxOptim():
    def __init__(self, data, gamma_inv, params, track=False):
        self.data = data
        self.gamma_inv = gamma_inv
        self.params = params
        self.nz = params['nonzero_inds']
        self.Nnz = self.nz.size
        self.track = track
        if 'zs_flattened' in params.keys():
            self.zs_flattened = params['zs_flattened']
        else:
            self.zs_flattened = False
        
        self.cost_func = get_e_step_cost_func(data, gamma_inv, params, self.zs_flattened)
        self.cost_grad = jax.grad(self.cost_func, holomorphic=True)
        self.cost_hess = jax.hessian(self.cost_func, holomorphic=True)        

    def eval_cost(self, zs):
        cost = self.cost_func(zs)
        grad = self.cost_grad(zs)
        hess = self.cost_hess(zs)
        hess_sel = jnp.stack([hess[n,:,n,:] for n in range(self.Nnz)])

        return cost, grad.conj(), hess_sel

    def run_e_step(self, zs_init, num_iters):
        zs_est = zs_init
        if self.track is True:
            track_zs = [zs_init]
            track_cost = []
            track_grad = []
            track_hess = []

        for _ in range(num_iters):
            cost, grad, hess = self.eval_cost(zs_est)
            hess_inv = jnp.linalg.inv(hess)

            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_inv, grad)
            if self.track is True:
                track_zs.append(zs_est)
                track_cost.append(cost)
                track_grad.append(grad) 
                track_hess.append(hess) 


        if self.track is True:
            result = OptimResult(zs_est, hess, track_zs, track_cost, track_grad, track_hess)
        else:
            result = OptimResult(zs_est, hess)
        self.result = result


def conv_grad_old_r2c(grad_vec_real, K):
    rs = grad_vec_real.reshape(2,-1).swapaxes(0,1)
    grad_vec_complex = conv_v_to_z(rs, axis=0)
    return grad_vec_complex

conv_mus_old_r2c = conv_grad_old_r2c


class OldOptim():
    def __init__(self, data, gamma_inv, params, track=False):
        self.data = data
        self.gamma_inv = gamma_inv
        self.params = params
        self.track = track

        self.obs_var = params['obs_var']
        self.Wv = params['Wv']
        self.num_J_vars = self.Wv.shape[1]
        self.K = data.shape[1]

        nz = params['nonzero_inds']
        sample_length = self.Wv.shape[0]

        invQ = jnp.diag(jnp.ones(sample_length)*(1/self.obs_var))

        obs_objs = [GaussianTrial(data[None,:,i], invQ) for i in range(self.K)] 
        gamma_inv_oldformat = construct_Gamma_full_real(self.gamma_inv[nz,:,:], self.K, self.num_J_vars, invert=False)
        trial_obj = TrialDataGaussian(obs_objs, gamma_inv_oldformat, self.Wv)

        self.cost_func = trial_obj.cost_func()
        self.cost_grad = trial_obj.cost_grad()
        self.cost_hess = trial_obj.cost_hess()

        self.optim_result = None

    def eval_cost(self, zs):
        vs = conv_z_to_v(zs, axis=0)
        vs_flat = jnp.concatenate([vs[:,k] for k in range(self.K)])

        cost_real = self.cost_func(vs_flat)
        grad_real = self.cost_grad(vs_flat)
        hess_full_real = self.cost_hess(vs_flat)

        grad = conv_grad_old_r2c(grad_real, self.K)
        hess = deconstruct_Gamma_full_real(hess_full_real, self.K, self.num_J_vars)

        return cost_real, grad, hess

    def run_e_step(self, zs_init, num_iters):
        zs_est = zs_init
        if self.track is True:
            track_zs = [zs_init]
            track_cost = []
            track_grad = []
            track_hess = []

        for _ in range(num_iters):
            cost, grad, hess = self.eval_cost(zs_est)
            hess_inv = jnp.linalg.inv(hess)

            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_inv, grad)
            if self.track is True:
                track_zs.append(zs_est)
                track_cost.append(cost)
                track_grad.append(grad) 
                track_hess.append(hess) 


        if self.track is True:
            result = OptimResult(zs_est, hess, track_zs, track_cost, track_grad, track_hess)
        else:
            result = OptimResult(zs_est, hess)
        self.result = result

class OldOptimMod():
    def __init__(self, data, gamma_inv, params, track=False):
        self.data = data
        self.gamma_inv = gamma_inv
        self.params = params
        self.track = track

        self.obs_var = params['obs_var']
        self.Wv = params['Wv']
        self.num_J_vars = self.Wv.shape[1]
        self.K = data.shape[1]

        nz = params['nonzero_inds']
        sample_length = self.Wv.shape[0]

        invQ = jnp.diag(jnp.ones(sample_length)*(1/self.obs_var))

        obs_objs = [GaussianTrial(data[None,:,i], invQ) for i in range(self.K)] 
        # obs_objs = [GaussianTrialMod(data[None,:,i], invQ) for i in range(self.K)] 
        # gamma_inv_oldformat = construct_Gamma_full_real(self.gamma_inv[nz,:,:], self.K, self.num_J_vars, invert=False)
        gamma_inv_oldformat = construct_Gamma_full_real_mod(self.gamma_inv[nz,:,:], self.K, self.num_J_vars, invert=False)
        trial_obj = TrialDataGaussian(obs_objs, gamma_inv_oldformat, self.Wv)

        self.cost_func = trial_obj.cost_func()
        self.cost_grad = trial_obj.cost_grad()
        self.cost_hess = trial_obj.cost_hess()

        self.optim_result = None

    def eval_cost(self, zs):
        vs = conv_z_to_v(zs, axis=0)
        vs_flat = jnp.concatenate([vs[:,k] for k in range(self.K)])

        cost_real = self.cost_func(vs_flat)
        grad_real = self.cost_grad(vs_flat)
        hess_full_real = self.cost_hess(vs_flat)

        grad = conv_grad_old_r2c(grad_real, self.K)
        # hess = deconstruct_Gamma_full_real(hess_full_real, self.K, self.num_J_vars)
        hess = 2*deconstruct_Gamma_full_real_mod(hess_full_real, self.K, self.num_J_vars)

        return cost_real, grad, hess

    def run_e_step(self, zs_init, num_iters):
        zs_est = zs_init
        if self.track is True:
            track_zs = [zs_init]
            track_cost = []
            track_grad = []
            track_hess = []

        for _ in range(num_iters):
            cost, grad, hess = self.eval_cost(zs_est)
            hess_inv = jnp.linalg.inv(hess)

            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_inv, grad)
            if self.track is True:
                track_zs.append(zs_est)
                track_cost.append(cost)
                track_grad.append(grad) 
                track_hess.append(hess) 


        if self.track is True:
            result = OptimResult(zs_est, hess, track_zs, track_cost, track_grad, track_hess)
        else:
            result = OptimResult(zs_est, hess)
        self.result = result

class JvOExp():
    def __init__(self, obs, gamma_inv, obs_var, params, method, track=False):
        self.obs = obs
        self.gamma_inv = gamma_inv
        self.params = params
        self.params['obs_var'] = obs_var
        self.method = method
        self.Nnz = params['nonzero_inds'].size
        self.K = obs.shape[1]
        self.track = track
        self.track_data = {}


    def eval_cost(self, trial, zs=None):
        if zs is None:
            zs = jnp.zeros((self.Nnz, self.K), dtype=complex)

        trial_data = self.obs[:,:,trial]
        if self.method == 'jax':
            optimizer = JaxOptim
        elif self.method == 'old':
            optimizer = OldOptim
        elif self.method == 'oldmod':
                optimizer = OldOptimMod
        else:
            raise ValueError

        optim = optimizer(trial_data, self.gamma_inv, self.params)
        cost, grad, hess = optim.eval_cost(zs)
        return cost, grad, hess

    def e_step(self, num_iters, zs_init=None):
        L = self.obs.shape[-1]


        if zs_init is None:
            Nnz = self.params['nonzero_inds'].size
            K = self.obs.shape[1]
            zs_init = jnp.zeros((Nnz, K), dtype=complex)

        mus = jnp.zeros((self.Nnz,K,L), dtype=complex)
        Upss = jnp.zeros((self.Nnz,K,K,L), dtype=complex)

        for trial in tqdm(range(L)):
            trial_data = self.obs[:,:,trial]
            if self.method == 'jax':
                optimizer = JaxOptim
            elif self.method == 'old':
                optimizer = OldOptim
            elif self.method == 'oldmod':
                optimizer = OldOptimMod
            else:
                raise ValueError

            optim = optimizer(trial_data, self.gamma_inv, self.params, self.track)
            optim.run_e_step(zs_init, num_iters)
            if self.track is True:
                self.track_data[trial] = optim.result

            mus = mus.at[:,:,trial].set(optim.result.zs_est)
            Upss = Upss.at[:,:,:,trial].set(optim.result.hess)

        self.mus = mus
        self.Upss = Upss

def construct_Gamma_full_real_mod(Gamma, K, num_J_vars, invert=False):
    return construct_Gamma_full_real(Gamma, K, num_J_vars, invert=invert)*4

def deconstruct_Gamma_full_real_mod(Gamma, K, num_J_vars, invert=False):
    return deconstruct_Gamma_full_real(Gamma, K, num_J_vars, invert=invert)/4