from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import LowRankCCN

@dataclass
class BasicSingleFreq:
    latent_type: str = 'basic_single'
    K: int = 3
    rank: int = 1
    num_freqs: int = 500 # assuming 1 second window 
    # target_freq_inds: List[int] = field(default_factory=lambda: [9])
    target_freq_ind: int = 9
    scale_power_target: float = 1.0e3
    L: int = 50
    seed: int = 7 

@dataclass
class BasicSingleFreqReLU:
    """
    Scaled version of BasicSingleFreq for point-process observations w/ ReLU Link
    """
    latent_type: str = 'basic_single_relu'
    K: int = 3
    rank: int = 1
    num_freqs: int = 500 # assuming 1 second window 
    # target_freq_inds: List[int] = field(default_factory=lambda: [9])
    target_freq_ind: int = 9
    scale_power_target: float = 1.0e3*(3*1.0e4)
    L: int = 50
    seed: int = 0 

@dataclass
class BasicSingleFreqLog:
    """
    Scaled version of BasicSingleFreq for point-process observations w/ Log Link
    """
    latent_type: str = 'basic_single_log'
    K: int = 3
    rank: int = 1
    num_freqs: int = 500 # assuming 1 second window 
    # target_freq_inds: List[int] = field(default_factory=lambda: [9])
    target_freq_ind: int = 9
    scale_power_target: float = 1.0e6
    L: int = 50
    seed: int = 7 

def create_lrccn_basic_rank1(lcfg):
    K = lcfg.K
    N = lcfg.num_freqs
    target_freq_ind = lcfg.target_freq_ind
    seed = lcfg.seed
    freqs = jnp.arange(N)
    R = lcfg.rank

    scale_target = lcfg.scale_power_target

    nz = jnp.array([target_freq_ind])

    Nnz = nz.size

    eigvecs_target = jnp.zeros((Nnz,K,R), dtype=complex)
    eigvals_target = jnp.zeros((Nnz,R))

    lrk = jr.key(seed)
    rksplit = jr.split(lrk, Nnz*R)
    for j in range(Nnz):
        for r in range(R):
            if Nnz == 1 and R == 1:
                rk = lrk
            else:
                rk = rksplit[j+r]
            phases = jr.uniform(rk, (K,), minval=-jnp.pi, maxval=jnp.pi)
            reals = jnp.cos(phases)
            imags = jnp.sin(phases)

            eigvec = reals + 1j*imags
            eigvec = eigvec / jnp.linalg.norm(eigvec)
            eigvec = eigvec*jnp.exp(-1j*jnp.angle(eigvec[0]))

            eigvecs_target = eigvecs_target.at[j,:,r].set(eigvec)

            eigval = K*scale_target 
            eigvals_target = eigvals_target.at[j,r].set(eigval)

    lrccn = LowRankCCN(eigvals_target, eigvecs_target, K, freqs, nz)
    return lrccn