from dataclasses import dataclass, field
from typing import List, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


from .app import AppRat15
from .latent import BasicSingleFreq, BasicSingleFreqLog, BasicSingleFreqReLU, AppFullRankSingleFreq
from .obs import GaussianObs, PPLogObs, PPReluObs, AppPPLogObs
from .model import LowRankToySimpleM1, FullRankToySimple, FullRankToyPseudoInv

def register_app_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='app', name='appdata_rat15', node=AppRat15)
    cs.store(group='latent', name='app_fullrank_single_freq', node=AppFullRankSingleFreq)
    cs.store(group='obs', name='app_pp_log', node=AppPPLogObs)
    cs.store(group='model', name='lowrank_eigh', node=LowRankToySimpleM1)
    cs.store(group='model', name='fullrank', node=FullRankToySimple)
    cs.store(group='model', name='fullrank_pinv', node=FullRankToyPseudoInv)

def get_app_config():
    register_app_configs()

    defaults = [
        {"app": "appdata_rat15"},
        {"latent": "app_fullrank_single_freq"},
        {"obs": "app_pp_log"},
        {"model": "fullrank_pinv"}
    ]

    @dataclass 
    class AppConfig:
        defaults: List[Any] = field(default_factory=lambda: defaults)
        app: Any = MISSING
        latent: Any = MISSING
        obs: Any = MISSING
        model: Any = MISSING

    cs = ConfigStore.instance()
    cs.store("config", node=AppConfig)
    
    return AppConfig

def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group='latent', name='single_freq', node=BasicSingleFreq)
    cs.store(group='latent', name='single_freq_log', node=BasicSingleFreqLog)
    cs.store(group='latent', name='single_freq_relu', node=BasicSingleFreqReLU)
    cs.store(group='obs', name='gaussian', node=GaussianObs)
    cs.store(group='obs', name='pp_log', node=PPLogObs)
    cs.store(group='obs', name='pp_relu', node=PPReluObs)
    cs.store(group='model', name='lowrank_eigh', node=LowRankToySimpleM1)
    cs.store(group='model', name='fullrank', node=FullRankToySimple)
    cs.store(group='model', name='fullrank_pinv', node=FullRankToyPseudoInv)


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
        {"latent": "single_freq_log"},
        {"obs": "pp_log"},
        # {"model": "lowrank_eigh"}
        {"model": "fullrank"}
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

