from dataclasses import dataclass, field
from typing import List, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


from .latent import BasicSingleFreq, BasicSingleFreqLog, BasicSingleFreqReLU
from .obs import GaussianObs, PPLogObs, PPReluObs
from .model import LowRankToySimpleM1

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='latent', name='single_freq', node=BasicSingleFreq)
    cs.store(group='latent', name='single_freq_log', node=BasicSingleFreqLog)
    cs.store(group='latent', name='single_freq_relu', node=BasicSingleFreqReLU)
    cs.store(group='obs', name='gaussian', node=GaussianObs)
    cs.store(group='obs', name='pp_log', node=PPLogObs)
    cs.store(group='obs', name='pp_relu', node=PPReluObs)
    cs.store(group='model', name='lr_eigh', node=LowRankToySimpleM1)


def get_sim_config():
    register_configs()

    defaults = [
        {"latent": "single_freq_log"},
        {"obs": "pp_log"}
    ]

    @dataclass 
    class SimConfig:
        defaults: List[Any] = field(default_factory=lambda: defaults)
        latent: Any = MISSING
        obs: Any = MISSING

    cs = ConfigStore.instance()
    cs.store("config", node=SimConfig)

    return SimConfig

def get_fit_config():
    register_configs()

    defaults = [
        {"latent": "single_freq"},
        {"obs": "gaussian"},
        {"model": "lr_eigh"}
    ]

    @dataclass 
    class FitConfig:
        defaults: List[Any] = field(default_factory=lambda: defaults)
        latent: Any = MISSING
        obs: Any = MISSING
        model: Any = MISSING
    
    return FitConfig

