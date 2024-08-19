import jax.numpy as jnp
import jax.random as jr

# TODO 
# generalize
# - create generic 'ccn_sample' function and use

# Generics
def sample_ccn(rk, cov, L):
    """
    Generate L samples from K-dimensional multivariate complex 
    normal (circular symmetric). 'cov' must be psd Hermitian.

    Args:
        rk: jax random key
        cov: (K,K) array PSD Hermitian covariance matrix
        L: number of samples

    Returns:
        samples: (K,L) array (complex)
    
    """
    assert jnp.all(cov.conj().T == cov)
    K = cov.shape[0]

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    assert jnp.all(eigvals >= 0)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples

# Application
def sample_from_gamma(rk, gamma, L):
    """
    Generate L samples from multivariate complex normal (circular symmetric). 
    Gamma is assumed to be indepedent across frequencies (block-diagonal).

    Args:
        gamma: (N,K,K) array where N is frequencies in nyquist 
            range, K is number of units
        L: number of samples to generate
        rk: jax random key
    Return:
        samples: (N,K) array (complex)
    """
    N = gamma.shape[0]

    rksplit = jr.split(rk,N)
    samples = jnp.stack([sample_ccn(rksplit[n], gamma[n,:,:], L) for n in range(N)])
    # samples = jnp.stack([sample_ccn(rk+n, gamma[n,:,:], L) for n in range(N)])

    return samples