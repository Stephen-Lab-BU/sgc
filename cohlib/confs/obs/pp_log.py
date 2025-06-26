from dataclasses import dataclass

@dataclass
class PPLogObs:
    obs_type: str = 'pp_log'
    mu: float = 1.9
    delta: float = 0.001
    seed: int = 42

@dataclass
class AppPPLogObs:
    obs_type: str = 'app_pp_log'
    mu_option: str = 'empirical'
    delta: float = 0.001