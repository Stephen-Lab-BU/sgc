from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr

from cohlib.latent import LowRankCCN, CCN
from cohlib.utils import rotate_eigvecs

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
    gamma_seed: int = 0
    seed: int = 7 

@dataclass
class BasicSingleFreqReLU(BasicSingleFreq):
    """
    Scaled version of BasicSingleFreq for point-process observations w/ ReLU Link
    """
    latent_type: str = 'basic_single_relu'
    scale_power_target: float = 1.0e3*(3*1.0e4)

@dataclass
class BasicSingleFreqLog(BasicSingleFreq):
    """
    Scaled version of BasicSingleFreq for point-process observations w/ Log Link
    """
    latent_type: str = 'basic_single_log'
    scale_power_target: float = 1.0e6

def create_lrccn_basic_rank1(lcfg, print_seed=False):
    K = lcfg.K
    N = lcfg.num_freqs
    target_freq_ind = lcfg.target_freq_ind
    gamma_seed = lcfg.gamma_seed
    freqs = jnp.arange(N)
    R = lcfg.rank

    scale_target = lcfg.scale_power_target

    nz = jnp.array([target_freq_ind])

    Nnz = nz.size

    eigvecs_target = jnp.zeros((Nnz,K,R), dtype=complex)
    eigvals_target = jnp.zeros((Nnz,R))

    if print_seed:
        print(f"Creating equal magnitude eigenvector with random phases (gamma seed: {gamma_seed})")

    gamma_rk = jr.key(gamma_seed)
    rksplit = jr.split(gamma_rk, Nnz*R)
    for j in range(Nnz):
        for r in range(R):
            if Nnz == 1 and R == 1:
                rk = gamma_rk
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

def create_ccn_basic_fullrank(lcfg, print_seed=True):

    assert lcfg.K == lcfg.rank
    K = lcfg.K
    gamma_seed = lcfg.gamma_seed
    N = lcfg.num_freqs
    target_freq_ind = lcfg.target_freq_ind
    gamma_seed = lcfg.gamma_seed

    nz = jnp.array([target_freq_ind])

    freqs = jnp.arange(N)
    gamma_seed = 3
    Nnz = 1

    if print_seed:
        print(f"Creating equal magnitude eigenvector with random phases (gamma seed: {gamma_seed})")

    gamma = jnp.zeros((Nnz, K, K), dtype=complex)
    gamma_rk = jr.key(gamma_seed)
    rksplit = jr.split(gamma_rk, Nnz)
    for j in range(Nnz):
        if Nnz == 1: 
            rk = gamma_rk
        else:
            rk = rksplit[j]

        rksplit2 = jr.split(rk, 2)
        Q, _ = jnp.linalg.qr(jr.normal(rksplit2[0], (K,K)) + jr.normal(rksplit2[1], (K,K))*1j)

        Q = rotate_eigvecs(Q[None,:,:])

        gamma_j = Q[j,:,:] @ jnp.diag(jnp.array([1e6, 1e3, 1e2])*K) @ Q[j,:,:].conj().T
        gamma = gamma.at[j,:,:].set(gamma_j)

    ccn = CCN(gamma, freqs, nz)
    return ccn