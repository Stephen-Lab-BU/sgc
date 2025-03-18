import jax.numpy as jnp
import jax.random as jr

# TODO add abc
class LowRankCCN():
    def __init__(self, eigvals, eigvecs, dim, freqs, nonzero_inds):
        self.freqs = freqs
        self.N = freqs.size
        self.nz = nonzero_inds
        self.Nnz = nonzero_inds.size
        self.rank = eigvals.shape[1]
        self.dim = dim
        self.eigvals = eigvals
        self.eigvecs = eigvecs

    def get_gamma(self):
        gamma = jnp.zeros((self.Nnz, self.dim, self.dim), dtype=complex)
        U_blank = jnp.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.Nnz):
            eigvals_j_lr = self.eigvals[j,:]
            eigvals_j = jnp.zeros(self.dim)
            L = jnp.diag(eigvals_j.at[:self.rank].set(eigvals_j_lr))

            eigvecs_j_lr = self.eigvecs[j,:,:]

            U = U_blank.copy()
            U = U.at[:,:self.rank].set(eigvecs_j_lr)

            gamma = gamma.at[j,:,:].set(U @ L @ U.conj().T)

        return gamma

    def get_gamma_pinv(self):
        gamma_pinv = jnp.zeros((self.Nnz, self.dim, self.dim), dtype=complex)
        U_blank = jnp.zeros((self.dim, self.dim), dtype=complex)
        for j in range(self.Nnz):
            eigvals_j_lr = 1 / self.eigvals[j,:]

            # If hard setting an eigval to 0
            eigvals_j_lr = jnp.nan_to_num(eigvals_j_lr,posinf=0,neginf=0)
            eigvals_j = jnp.zeros(self.dim, dtype=complex)
            Lam = jnp.diag(eigvals_j.at[:self.rank].set(eigvals_j_lr))

            eigvecs_j_lr = self.eigvecs[j,:,:]

            U = U_blank.copy()
            U = U.at[:,:self.rank].set(eigvecs_j_lr)

            gamma_pinv = gamma_pinv.at[j,:,:].set(U @ Lam @ U.conj().T)

        return gamma_pinv

    def sample_nz(self, rk, L):
        if self.Nnz == 1:
            samples_nz = sample_lrccn(rk, self.eigvecs[0,:,:], self.eigvals[0,:], self.dim, L)
            samples_nz = samples_nz[None,:,:]
        else:
            rksplit = jr.split(rk, self.Nnz)
            samples_nz = jnp.stack([sample_lrccn(rksplit[n], self.eigvecs[n,:,:], self.eigvals[n,:], 
                                self.dim, L) for n in range(self.Nnz)])
        return samples_nz

    def sample(self, rk, L):
        samples_nz = self.sample_nz(rk, L)
        samples = jnp.zeros((self.N,self.dim,L),dtype=complex)
        samples = samples.at[self.nz,:,:].set(samples_nz)
        return samples

class CCN():
    def __init__(self, gamma, freqs, nonzero_inds, inv_flag='standard'):
        self.freqs = freqs
        self.N = freqs.size
        self.nz = nonzero_inds
        self.Nnz = nonzero_inds.size
        self.dim = gamma.shape[-1]
        self.rank = gamma.shape[-1]
        self.gamma = gamma
        self.inv_flag = inv_flag

    def get_gamma(self):
        return self.gamma

    def get_gamma_inv(self):
        gamma_inv = jnp.zeros((self.Nnz, self.dim, self.dim), dtype=complex)
        for j in range(self.Nnz):
            if self.inv_flag == 'standard':
                gamma_inv = gamma_inv.at[j,:,:].set(jnp.linalg.inv(self.gamma[j,:,:]))
            elif self.inv_flag == 'pinv':
                gamma_inv = gamma_inv.at[j,:,:].set(jnp.linalg.pinv(self.gamma[j,:,:]))
            else:
                raise ValueError

        return gamma_inv

    def sample_nz(self, rk, L):
        if self.Nnz == 1:
            samples_nz = sample_ccn(rk, self.gamma[0,:,:], L)
            samples_nz = samples_nz[None,:,:]
        else:
            rksplit = jr.split(rk, self.Nnz)
            samples_nz = jnp.stack([sample_ccn(rksplit[n], self.gamma[n,:,:], 
                                L) for n in range(self.Nnz)])
        return samples_nz

    def sample(self, rk, L):
        samples_nz = self.sample_nz(rk, L)
        samples = jnp.zeros((self.N,self.dim,L),dtype=complex)
        samples = samples.at[self.nz,:,:].set(samples_nz)
        return samples

# TODO add references for why this works
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
    assert jnp.all(jnp.isclose(cov.conj().T, cov, atol=1e-9))
    K = cov.shape[0]

    eigvals, eigvecs = jnp.linalg.eigh(cov)
    # assert jnp.all(eigvals >= 0)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples

def sample_lrccn(rk, eigvecs_lr, eigvals_lr, K, L):
    R = eigvals_lr.size

    eigvecs = jnp.zeros((K,K), dtype=complex)
    eigvecs = eigvecs.at[:,:R].set(eigvecs_lr)

    eigvals = jnp.zeros(K)
    eigvals = eigvals.at[:R].set(eigvals_lr)

    D = jnp.diag(jnp.sqrt(eigvals))
    A = eigvecs @ D

    unit_samples = jr.normal(rk, (K,L), dtype=complex)
    samples = jnp.einsum('ki,il->kl', A, unit_samples)

    return samples