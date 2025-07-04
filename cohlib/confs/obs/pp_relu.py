from dataclasses import dataclass

@dataclass
class PPReluObs:
    obs_type: str = 'pp_relu'
    mu: float = 100.0
    delta: float = 0.001
    seed: int = 0