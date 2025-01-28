from dataclasses import dataclass

@dataclass
class LowRankToySimpleM1:
    model_type: str = 'simple_inherit_latent_eigh'
    model_rank: int = 1
    inherit_lcfg: bool = True # window / non-zero frequencies same as generating data
    num_em_iters: int = 20
    num_newton_iters: int = 10
    m_step_option: str = 'low-rank-eigh'
    eigvecs_flag: str = 'fit'
    eigvals_flag: str = 'oracle'
    scale_init: float = 1.0
    model_init: str = 'oracle'
