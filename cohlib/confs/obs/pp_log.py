from dataclasses import dataclass

@dataclass
class PPLogObs:
    obs_type: str = 'pp_log'
    mu: float = 1.9
    delta: float = 0.001
    seed: int = 42