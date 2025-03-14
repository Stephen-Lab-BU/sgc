import os
import multiprocessing
from functools import partial 

import jax
import jax.numpy as jnp

def add_dc(x, dc):
    dc_arr = jnp.array([dc])
    with_dc = jnp.concatenate([dc_arr, x])
    return with_dc
add0 = partial(add_dc, dc=0)

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
    R = eigvecs.shape[2]

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