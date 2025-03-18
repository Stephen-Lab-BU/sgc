from dataclasses import dataclass

@dataclass
class GaussianObs:
    obs_type: str = 'gaussian'
    ov1: float = 1.0
    ov2: float = -3.0
    seed: int = 7