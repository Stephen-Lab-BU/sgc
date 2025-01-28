import os
import shutil 
from pathlib import Path
from omegaconf import OmegaConf
import jax.numpy as jnp

from cohlib.utils import  pickle_open

# TODO - deprecate? this code was used when all runs were saved into a single
# folder and we had to go through and load each config to check if it was the
# one we wanted. could still be useful so not deleting yet.

def check_attrs(cfg, lcfg_attrs, mcfg_attrs, ocfg_attrs):    
    lcfg_check = [cfg.latent.get(k, None) == v for k, v in lcfg_attrs.items()]
    mcfg_check = [cfg.model.get(k, None) == v for k, v in mcfg_attrs.items()]
    ocfg_check = [cfg.obs.get(k, None) == v for k, v in ocfg_attrs.items()]

    check = jnp.all(jnp.array(lcfg_check + mcfg_check + ocfg_check))
    return check

def filter_load_results(paths, lcfg_attrs, mcfg_attrs, ocfg_attrs):
    sel_paths = []
    for path in paths:
        _dir = Path(path)
        for i, exp in enumerate(_dir.glob('*')):
            cfg_path = os.path.join(exp, '.hydra/config.yaml')
            cfg = OmegaConf.load(cfg_path)

            if check_attrs(cfg, lcfg_attrs, mcfg_attrs, ocfg_attrs):
                sel_paths.append(exp)

    assert len(sel_paths) > 0
    print(f'{len(sel_paths)} paths found meeting critera')

    results = []
    for path in sel_paths:
        cfg_path = os.path.join(path, '.hydra/config.yaml')
        cfg = OmegaConf.load(cfg_path)

        res = pickle_open(os.path.join(path, 'res.pickle'))
        res['cfg'] = cfg
        results.append(res)

    return results


def filter_loaded(loaded, lcfg_attrs, mcfg_attrs, ocfg_attrs):
    filtered = [r for r in loaded if check_attrs(r['cfg'], lcfg_attrs, mcfg_attrs, ocfg_attrs)]
    if len(filtered) == 0:
        print("No results in list to filter.")
    elif len(filtered) == 1:
        print("Returned single result.")
        return filtered[0]
    else:
        print("Multiple results found.")
        return filtered

def remove_duplicates(path):
    print(f'Searching for duplicate configs in directory {path}')
    _dir = Path(path)
    exp_times = []
    config_list = []
    for i, exp in enumerate(_dir.glob('*')):
        exp_times.append(os.path.split(str(exp))[1])
        config_path = os.path.join(exp, '.hydra/config.yaml')
        config = OmegaConf.load(config_path)
        config_list.append(config)

        # if check_attrs(cfg, lcfg_attrs, mcfg_attrs, ocfg_attrs):
        #     sel_paths.append(exp)

    dupe_inds = []
    for entry in range(len(config_list)):
        for i in range(entry+1, len(config_list)):
            if str(config_list[entry]) == str(config_list[i]):
                dupe_inds.append(i)

    dupe_inds = jnp.unique(jnp.array(dupe_inds))

    for dupe_ind in dupe_inds:
        exp_time = exp_times[dupe_ind]
        del_path = os.path.join(path, exp_time)
        shutil.rmtree(del_path)
    print(f'{dupe_inds.size} duplicate configs found in directory removed.')