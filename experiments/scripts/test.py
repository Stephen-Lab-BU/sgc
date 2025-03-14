import os
import hydra
from omegaconf import OmegaConf
import jax.numpy as jnp

from cohlib.utils import pickle_open
import cohlib.confs.utils as conf
from cohlib.jax.dists import LowRankCCN, CCN
from cohlib.jax.lr_model import LowRankToyModel
from cohlib.jax.general_model import GeneralToyModel
from cohlib.confs.config import register_configs
from cohlib.jax.utils import jax_boilerplate

from dataclasses import dataclass, field
from typing import List, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

jax_boilerplate()

# Create config - (incorporates overrides from from command line)

def get_fit_config():
    register_configs()

    defaults = [
        {"latent": "single_freq_log"},
        {"obs": "pp_log"},
        {"model": "lowrank_eigh"}
    ]

    @dataclass 
    class FitConfig:
        defaults: List[Any] = field(default_factory=lambda: defaults)
        latent: Any = MISSING
        obs: Any = MISSING
        model: Any = MISSING

    cs = ConfigStore.instance()
    cs.store("config", node=FitConfig)
    
    return FitConfig

Config = get_fit_config()

@hydra.main(version_base=None, config_name = "config")
def run(cfg: Config) -> None:
    run_path = conf.get_run_path()
    os.chdir(run_path)

    print('Fitting model from config:')
    print(OmegaConf.to_yaml(cfg))

    seed = 0
    L = 4
    mu = -1.0

    evalf = 'fit'
    init = 'empirical'

    lcfg = cfg.latent
    ocfg = cfg.obs
    mcfg = cfg.model

    lcfg.L = L
    lcfg.seed = seed
    ocfg.mu = mu
    ocfg.seed = seed
    mcfg.eigvals_flag = evalf
    mcfg.model_init = init

    latent_dir = conf.get_latent_dir(lcfg)
    obs_dir = conf.get_obs_dir(ocfg, latent_dir)
    model_subdir = conf.get_model_subdir(mcfg)
    model_dir = os.path.join(obs_dir, model_subdir)

    # Check that model is not already saved at savepath.
    try:
        model_load = pickle_open(os.path.join(model_dir, 'res.pkl'))
        saved_cfg = conf.omega(model_load['cfg'])
        if cfg == saved_cfg:
            print("Saved model with identical config found. Model run aborted.")
        else:
            print(f"Model save location is: {model_dir}")
            raise AssertionError("Model already saved in location. Saved config conflicts with config for current run. Model run aborted.")


    except FileNotFoundError:
        print('Loading data and running model.')

        # configure parameters
        load_latent = pickle_open(os.path.join(latent_dir, 'latent_sim.pkl'))
        load_obs = pickle_open(os.path.join(obs_dir, 'obs_sim.pkl'))
        obs_params, obs_type = conf.get_obs_params(ocfg)
        zs_nz = load_latent['zs_nz']
        obs = load_obs['obs']

        if mcfg.inherit_lcfg:
            K = lcfg.K
            N = lcfg.num_freqs
            freqs = jnp.arange(N)
            nz_model = jnp.array([lcfg.target_freq_ind])

            init_params = {
                        'nz_model': nz_model,
                        'scale_init': mcfg.scale_init,
                        'K': K,
                        'freqs': freqs,
                        'lcfg': lcfg,
                        'zs_nz': zs_nz,
                        'ocfg': ocfg,
                        'obs': obs}
            if mcfg.model_type == 'simple_inherit_latent_lowrank_eigh':
                init_params['rank'] = mcfg.model_rank
            elif mcfg.model_type == 'simple_inherit_latent_fullrank':
                init_params['rank'] = K
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


        model, fit_params, ccn_init = get_model_object(mcfg, init_params, obs_type, obs_params)



def get_model_object(mcfg, init_params, obs_type, obs_params):

    fit_params = {'num_em_iters': mcfg.num_em_iters, 
                'num_newton_iters': mcfg.num_newton_iters,
                'm_step_option': mcfg.m_step_option}

    K = init_params['K']
    freqs = init_params['freqs']
    nz_model = init_params['nz_model']

    if mcfg.model_type == 'simple_inherit_latent_lowrank_eigh':

        eigvals_init, eigvecs_init = conf.create_lowrank_init_eigparams(mcfg.model_init, init_params)

        fixed_params = conf.get_fixed_params(mcfg.eigvals_flag, mcfg.eigvecs_flag, init_params)
        fit_params['fixed_params'] = fixed_params

        if 'eigvals' in fixed_params.keys():
            eigvals_init = fixed_params['eigvals']
        if 'eigvecs' in fixed_params.keys():
            eigvecs_init = fixed_params['eigvecs']

        ccn_init = LowRankCCN(eigvals_init, eigvecs_init, K, freqs, nz_model)

    elif mcfg.model_type == 'simple_inherit_latent_fullrank':
        gamma_init = conf.create_fullrank_gamma(mcfg.model_init, init_params)

        ccn_init = CCN(gamma_init, freqs, nz_model)

    else:
        raise NotImplementedError

    # model = GeneralToyModel()
    model = LowRankToyModel()

    model.initialize_latent(ccn_init)
    model.initialize_observations(obs_params, obs_type)

    return model, fit_params, ccn_init

if __name__ == "__main__":
    run()
