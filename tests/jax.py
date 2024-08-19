import jax.numpy as jnp
import jax.random as jr

from cohlib.jax.dists import sample_ccn, sample_from_gamma

# add ccn testing

def test_gamma_sample():
    rk = jr.key(7)
    L = 1000000
    K = 2
    N = 50

    S = jnp.array([[5,1],[1,5]],dtype=complex) + jnp.array([[0,-1],[1,0]],dtype=complex)*1j
    gamma_nonzero = jnp.stack([S, S.conj()+jnp.eye(2)*5, jnp.eye(2, dtype=complex)*2])

    gamma_full = jnp.zeros((N,K,K), dtype=complex)
    nz_inds = jnp.array([5, 12, 18])
    freqs = jnp.arange(1,51)
    nz_filt = jnp.isin(freqs, nz_inds)
    gamma_full = gamma_full.at[nz_filt,:,:].set(gamma_nonzero)

    zs = sample_from_gamma(rk, gamma_full, L)
    est = jnp.einsum('jil,jkl->jikl', zs, zs.conj()).mean(-1)
    assert jnp.all(jnp.isclose(est, gamma_full, atol=1e0))