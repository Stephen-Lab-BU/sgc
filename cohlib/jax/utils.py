from functools import partial 
import jax.numpy as jnp

def add_dc(x, dc):
    dc_arr = jnp.array([dc])
    with_dc = jnp.concatenate([dc_arr, x])
    return with_dc
add0 = partial(add_dc, dc=0)

