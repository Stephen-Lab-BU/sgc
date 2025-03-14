import os
import multiprocessing
import pickle
from functools import partial 

import jax
import jax.numpy as jnp

def pickle_open(file):
    with open(file, "rb") as handle:
        data = pickle.load(handle)
    return data


def pickle_save(data, save_name):
    with open(save_name, "wb") as handle:
        pickle.dump(data, handle)


def add_dc(x, dc):
    dc_arr = jnp.array([dc])
    with_dc = jnp.concatenate([dc_arr, x])
    return with_dc
add0 = partial(add_dc, dc=0)


def naive_estimator(spikes, nonzero_inds=None):
    "spikes has shape (time, unit, trial)"
    n_f0 = jnp.fft.rfft(spikes, axis=0)
    n_f = n_f0[1:,:,:]
    naive_est = jnp.einsum('jkl,jil->jkil', n_f, n_f.conj()).mean(-1)

    if nonzero_inds is None:
        return naive_est
    else:
        return naive_est[nonzero_inds, :, :]

def estimate_coherence(xf,yf, mag_sq=True):
    """
    Estimate coherence for a single frequency range from observed complex coefs. 
    Args:
        xf: (n_trials,) array of complex coefficients signal 1
        yf: (n_trials,) array of complex coefficients signal 2
        mag_sq: (bool) optional - return mean-squared coherence 
    Returns:
        coh: coherence estimate
    """

    Sxy = xf * yf.conj()
    Sxx = xf * xf.conj()
    Syy = yf * yf.conj()

    if mag_sq:
        num = jnp.abs(Sxy.mean(0))**2
        denom = Sxx.mean(0).real * Syy.mean(0).real

    else:
        num = jnp.abs(Sxy.mean(0))
        a = jnp.sqrt(Sxx.mean(0).real)
        b = jnp.sqrt(Syy.mean(0).real)
        denom = a*b

    coh  = num/denom

    return coh

def thr_coherence(Gamma, mag_sq=True):
    """
    Calculate theoretical coherence from covariance matrices. 
    Args:
        Gamma: (n_freqs, 2, 2) array of complex bcn covariance matrices
    Returns:
        t_coh: (n_freqs,) array of coherence values
    """

    if mag_sq:
        num = jnp.abs(Gamma[:,0,1])**2
        a = jnp.abs(Gamma[:,0,0])
        b = jnp.abs(Gamma[:,1,1])

    else:
        num = jnp.abs(Gamma[:,0,1])
        a = jnp.abs(Gamma[:,0,0])
        b = jnp.abs(Gamma[:,1,1])
        a, b = jnp.sqrt(a), jnp.sqrt(b)

    denom = a*b
    t_coh = num/denom

    return t_coh

def jax_boilerplate():
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"
    jax.config.update('jax_platform_name', 'cpu')
    platform = jax.lib.xla_bridge.get_backend().platform.casefold()
    print("Platform: ", platform)
    print(len(jax.devices()))

def rotate_eigvecs(eigvecs):
    """
    Args:
        eigvecs: J x K x R
    """

    J = eigvecs.shape[0]

    rotated = jnp.zeros_like(eigvecs)

    for j in range(J):
        thetas = jnp.angle(eigvecs[j,0,:])
        rotations = jnp.exp(-1j*thetas)

        rotated = rotated.at[j,:,:].set(eigvecs[j,:,:] * rotations[None,:])
    
    return rotated

def stdize_eigvecs(eigvecs):
    """
    Args:
        eigvecs: J x K x R
    """

    J = eigvecs.shape[0]
    R = eigvecs.shape[2]

    stdized = jnp.zeros_like(eigvecs)

    for j in range(J):
        for r in range(R):
            if eigvecs[j,0,r].real < 0:
                stdized = stdized.at[j,:,r].set(-eigvecs[j,:,r])
            else:
                stdized = stdized.at[j,:,r].set(eigvecs[j,:,r])
    
    return stdized