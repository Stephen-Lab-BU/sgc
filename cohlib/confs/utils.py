import os 
import jax.numpy as jnp
from omegaconf import OmegaConf

from cohlib.confs.latent.simple import create_lrccn_basic_rank1, create_ccn_basic_fullrank
from cohlib.confs.app.rat_ansesthesia import load_app_data
from cohlib.utils import naive_estimator

# Paths are structured as:
# Data/LatentType/Window/K/L/LatentSeed/ObsType/ObsParams/ObsSeed/
# with Data/LatentType/Window/K/L/LatentSeed containing pkl with zs
# and Data/LatentType/Window/K/L/LatentSeed/ObsType/ObsParams/ObsSeed/ containaing pkl with observations (and summary fig if produced)

def get_run_path():
    return '/projectnb/stephenlab/jtauber/cohlib/experiments'
def omega(cfg):
    return OmegaConf.create(cfg)

def get_latent_dir(lcfg):
    window = int(2*lcfg.num_freqs)
    if lcfg.K == lcfg.rank:
        rank_name = 'fullrank'
    else:
        rank_name = f'rank{lcfg.rank}'

    latent_dir = f'data/latent-{lcfg.latent_type}_{rank_name}/scale{int(lcfg.scale_power_target)}/window{window}/gseed{lcfg.gamma_seed}/K{lcfg.K}/L{lcfg.L}/lseed{lcfg.seed}'

    return latent_dir

def get_obs_dir(ocfg, latent_dir):
    if ocfg.obs_type == 'gaussian':
        obs_subdir = f'obs-{ocfg.obs_type}/ovb{ocfg.ov1}_ove{ocfg.ov2}/oseed{ocfg.seed}'
    elif ocfg.obs_type in ['pp_relu', 'pp_log']:
        obs_subdir = f'obs-{ocfg.obs_type}/mu{ocfg.mu}/oseed{ocfg.seed}'
    else:
        raise NotImplementedError

    obs_dir = os.path.join(latent_dir, obs_subdir)
    return obs_dir

def get_app_dir(acfg):
    app_dir = f'data/app/{acfg.data_name}/{acfg.exp_type}/'
    return app_dir

def get_app_latent_dir(lcfg, app_dir):
    window = int(2*lcfg.num_freqs)

    latent_subdir = f'latent-{lcfg.latent_type}/window{window}/K{lcfg.K}/L{lcfg.L}/'
    latent_dir = os.path.join(app_dir, latent_subdir)

    return latent_dir

def get_app_obs_dir(ocfg, latent_dir):
    if ocfg.obs_type == 'app_pp_log':
        obs_subdir = f'obs-{ocfg.obs_type}/mu-{ocfg.mu_option}/'
    else:
        raise NotImplementedError

    obs_dir = os.path.join(latent_dir, obs_subdir)
    return obs_dir

def get_app_obs_params(ocfg, obs):
    obs_type = ocfg.obs_type
    if obs_type == 'app_pp_log':
        if ocfg.mu_option == 'empirical':
            mu = get_app_mus_empirical(obs)
            obs_params = {'mu': mu, 'mu_type': 'vec', 'delta': ocfg.delta}
        else:
            raise NotImplementedError
    else:
        raise ValueError
    
    return obs_params, obs_type

def get_app_mus_empirical(obs):
    return jnp.log(1000*obs.mean((0,2)))

def get_model_subdir(mcfg):
    if mcfg.inherit_lcfg is True:
        if mcfg.model_type == 'simple_inherit_latent_fullrank' or mcfg.model_type == 'simple_inherit_latent_fullrank_pinv':
            if mcfg.model_init == 'flat':
                init_oom = jnp.log10(mcfg.scale_init)
                model_subdir =f'model-{mcfg.model_type}/inherit-{mcfg.inherit_lcfg}/m_step-{mcfg.m_step_option}/{mcfg.model_init}-init_{init_oom}/newton-{mcfg.num_newton_iters}_em-{mcfg.num_em_iters}'
            else:
                model_subdir =f'model-{mcfg.model_type}/inherit-{mcfg.inherit_lcfg}/m_step-{mcfg.m_step_option}/{mcfg.model_init}-init/newton-{mcfg.num_newton_iters}_em-{mcfg.num_em_iters}'
        elif mcfg.model_type == 'simple_inherit_latent_lowrank_eigh':
            model_subdir =f'model-{mcfg.model_type}/inherit-{mcfg.inherit_lcfg}/m_step-{mcfg.m_step_option}/{mcfg.model_init}-init_eigvals-{mcfg.eigvals_flag}_eigvecs-{mcfg.eigvecs_flag}/newton-{mcfg.num_newton_iters}_em-{mcfg.num_em_iters}'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return model_subdir

def get_model_dir(cfg, params):
    latent_dir = get_latent_dir(cfg.latent)
    obs_dir = get_obs_dir(cfg.obs, latent_dir)
    model_subdir = get_model_subdir(cfg.model)
    res_dir = os.path.join(obs_dir, model_subdir)
    return res_dir

def get_obs_params(ocfg):
    obs_type = ocfg.obs_type
    if obs_type in ['pp_relu', 'pp_log']:
        obs_params = {'mu': ocfg.mu, 'delta': ocfg.delta}
    elif obs_type == 'gaussian':
        obs_params = {'obs_var': ocfg.ov1 * 10**ocfg.ov2,}
    else:
        raise ValueError
    
    return obs_params, obs_type

def create_lowrank_init_eigparams(value_type, params):
    nz_model = params['nz_model']
    J = nz_model.size
    rank = params['rank']
    K = params['K']

    eigvals_init = jnp.zeros((J,rank))
    eigvecs_init = jnp.zeros((J,K,rank), dtype=complex)

    if value_type == 'true':
        lcfg = params['lcfg']
        # true init requires model / data rank match
        assert lcfg.rank == rank

        lrccn_true = create_lrccn_basic_rank1(lcfg)
        eigvals_true = lrccn_true.eigvals
        eigvecs_true = lrccn_true.eigvecs

        for j in range(J):
            eigvals_init = eigvals_init.at[j,:].set(eigvals_true[j,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_true[j,:,:])

    
    elif value_type == 'flat':
        scale_init = params['scale_init']

        ones_eigvals = jnp.ones_like(eigvals_init) * scale_init * K
        ones_eigvecs = jnp.ones_like(eigvecs_init)/K + 0*1j

        for j in range(J):
            eigvals_init = eigvals_init.at[j,:].set(ones_eigvals[j,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(ones_eigvecs[j,:,:])
        

    elif value_type == 'oracle':
        lcfg = params['lcfg']
        # true init requires model / data rank match
        assert lcfg.rank == rank

        zs_nz = params['zs_nz']
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs_nz, zs_nz.conj()).mean(-1)
        for j in range(J):
            freqind = nz_model[j]
            eigvals_oracle_j, eigvecs_oracle_j = jnp.linalg.eigh(gamma_oracle[freqind,:,:])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_oracle_j[:,-rank:][:,::-1])
            eigvals_init = eigvals_init.at[j,:].set(eigvals_oracle_j[-rank:][::-1])


    elif value_type == 'empirical':
        print('Using empirical (naive) estimate for initialization.')
        ocfg = params['ocfg']
        obs_type = ocfg.obs_type
        obs = params['obs']
        if obs_type in ['pp_relu', 'pp_log']:
            gamma_empirical = naive_estimator(obs, nz_model) * 1e6
        elif obs_type == 'gaussian':
            gamma_empirical = naive_estimator(obs, nz_model) 
        else:
            raise NotImplementedError
            
        for j, nz_ind in enumerate(nz_model):
            eigvals_empirical_j, eigvecs_empirical_j = jnp.linalg.eigh(gamma_empirical[j,:,:])
            eigvals_init = eigvals_init.at[j,:].set(eigvals_empirical_j[-rank:][::-1])
            eigvecs_init = eigvecs_init.at[j,:,:].set(eigvecs_empirical_j[:,-rank:][:,::-1])
    else:
        raise ValueError

    return eigvals_init, eigvecs_init

def create_fullrank_gamma(value_type, params):
    nz_model = params['nz_model']
    J = nz_model.size
    K = params['K']

    gamma_init = jnp.zeros((J,K,K), dtype=complex)

    if value_type == 'true':
        raise NotImplementedError
    
    elif value_type == 'flat':
        scale_init = params['scale_init']

        for j in range(J):
            gamma_init = gamma_init.at[j,:,:].set(jnp.eye(K) * scale_init)

    elif value_type == 'oracle':
        lcfg = params['lcfg']
        # true init requires model / data rank match
        zs_nz = params['zs_nz']
        gamma_oracle = jnp.einsum('jkl,jil->jkil', zs_nz, zs_nz.conj()).mean(-1)

        gamma_init = gamma_oracle

    elif value_type == 'empirical':
        print('Using empirical (naive) estimate for initialization.')
        ocfg = params['ocfg']
        obs_type = ocfg.obs_type
        obs = params['obs']
        if obs_type in ['pp_relu', 'pp_log', 'app_pp_log']:
            gamma_empirical = naive_estimator(obs, nz_model) * 1e6
        elif obs_type == 'gaussian':
            gamma_empirical = naive_estimator(obs, nz_model) 
        else:
            raise NotImplementedError
            
        gamma_init = gamma_empirical

    else:
        raise ValueError

    return gamma_init

def get_fixed_params(eigvals_flag, eigvecs_flag, params):
    fixed_params = {}

    if eigvals_flag != 'fit':
        fixed_type = eigvals_flag 
        fixed_eigvals, _ = create_lowrank_init_eigparams(fixed_type, params)
        fixed_params['eigvals'] = fixed_eigvals

    if eigvecs_flag != 'fit':
        fixed_type = eigvecs_flag 
        _, fixed_eigvecs = create_lowrank_init_eigparams(fixed_type, params)
        fixed_params['eigvecs'] = fixed_eigvecs
    
    return fixed_params