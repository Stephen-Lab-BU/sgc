from dataclasses import dataclass

@dataclass
class LowRankToySimpleM1:
    model_type: str = 'simple_inherit_latent_lowrank_eigh'
    model_rank: int = 1
    inherit_lcfg: bool = True # window / non-zero frequencies defined in lcfg
    num_em_iters: int = 20
    num_newton_iters: int = 10
    m_step_option: str = 'low-rank-eigh'
    eigvecs_flag: str = 'fit'
    eigvals_flag: str = 'oracle'
    scale_init: float = 1.0
    model_init: str = 'flat'

@dataclass
class FullRankToySimple:
    model_type: str = 'simple_inherit_latent_fullrank'
    inherit_lcfg: bool = True # window / non-zero frequencies defined in lcfg
    num_em_iters: int = 20
    num_newton_iters: int = 10
    m_step_option: str = 'full-rank-standard'
    scale_init: float = 1.0
    model_init: str = 'flat'

@dataclass
class FullRankToyPseudoInv:
    model_type: str = 'simple_inherit_latent_fullrank_pinv'
    inherit_lcfg: bool = True # window / non-zero frequencies defined in lcfg
    num_em_iters: int = 20
    num_newton_iters: int = 10
    m_step_option: str = 'full-rank-standard'
    scale_init: float = 1.0
    model_init: str = 'flat'
    inv_flag: str = 'pinv'
