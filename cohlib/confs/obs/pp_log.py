from dataclasses import dataclass

@dataclass
class PPLogObs:
    obs_type: str = 'pp_log'
    alpha: float = 1.0
    delta: float = 0.001
    seed: int = 7