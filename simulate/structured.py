from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from typing import Any, List

@dataclass
class BasicSingleFreq:
    K: int = 3
    num_freqs: int = 500
    # target_freq_inds: List[int] = field(default_factory=lambda: [9])
    target_freq_ind: int = 9
    scale_power_target: float = 1.0e3
    L: int = 50
    seed: int = 7 

@dataclass
class GaussianObs:
    obs_type: str = 'gaussian'
    ov1: int = 1
    ov2: int = -3
    seed: int = 7
    

defaults = [
    {"latent": "singlefreq"},
    {"observation": "gaussian"}
]

@dataclass 
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    latent: Any = MISSING
    observation: Any = MISSING

cs = ConfigStore.instance()
cs.store(group='latent', name='singlefreq', node=BasicSingleFreq)
cs.store(group='observation', name='gaussian', node=GaussianObs)
cs.store("config", node=Config )


@hydra.main(version_base=None, config_name = "config")
def run(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    create_gamma_basic_single(cfg.latent)


def create_gamma_basic_single(lcfg):
    K = lcfg.L
    N = lcfg.num_freqs
    target_freq_ind = lcfg.target_freq_ind

    # move ipynb code here to run as script
    # save dist object as output to be loaded in experiments
    # run plotting code and save figure showing overview 



if __name__ == "__main__":
    run()
