import abc
from typing import Any, Dict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

@dataclass
class ModelParams:
    """
    A dataclass to hold the model parameters.
    Everything inside should be a PyTree so that we can use
    jax utilities like grad, jit, etc.
    """
    # Example fields, replace with your model-specific parameters
    # For instance, means and variances for a Gaussian mixture model:
    # means: jnp.ndarray  # shape: [K, D]
    # covs: jnp.ndarray   # shape: [K, D, D]
    pass

@dataclass
class VariationalParams:
    """
    A dataclass to hold variational parameters for q(Z|X; phi).
    """
    # Example fields, depending on the variational family.
    # E.g., for a Gaussian variational approximation:
    # mean: jnp.ndarray   # shape: [N, K, D_latent]
    # logvar: jnp.ndarray # shape: [N, K, D_latent]
    pass

@dataclass
class PosteriorParams:
    """
    Parameters representing q(Z|X) from the E-step.
    This could be similar to variational params if q(Z|X) is known in closed form.
    """
    # Example fields for a distribution of latent variables:
    # responsibilities: jnp.ndarray # shape: [N, K] for a mixture model
    pass

class LatentVariableModel(abc.ABC):
    """
    A template for latent variable models, inspired by Dynamax-style functional design.
    Instead of storing state in the class, we define functions that take parameters as input
    and produce updated parameters as output. This aligns well with JAX's functional paradigm.
    """

    def __init__(self, model_config):
        self.model_config = model_config

    @abc.abstractmethod
    def init_params(self, config) -> ModelParams:
        """
        Initialize model parameters given data X.
        """
        pass

    @abc.abstractmethod
    def init_variational_params(self, X: jnp.ndarray) -> VariationalParams:
        """
        Initialize variational parameters for q(Z|X; phi).
        """
        pass

    @abc.abstractmethod
    def log_joint(self, params: ModelParams, X: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log p(X, Z).
        Should support batched Z.
        """
        pass

    @abc.abstractmethod
    def E_step(self, params: ModelParams, X: jnp.ndarray) -> PosteriorParams:
        """
        Compute posterior parameters q(Z|X) for the E-step of EM.
        """
        pass

    @abc.abstractmethod
    def M_step(self, qZ: PosteriorParams, X: jnp.ndarray) -> ModelParams:
        """
        Compute updated model parameters for the M-step of EM, given q(Z|X).
        """
        pass

    @abc.abstractmethod
    def sample_from_q(self, phi: VariationalParams, num_samples: int = 1) -> jnp.ndarray:
        """
        Sample latent variables Z from q(Z|X; phi).
        """
        pass

    @abc.abstractmethod
    def log_q(self, phi: VariationalParams, Z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log q(Z|X; phi).
        Should support batched Z.
        """
        pass

    def fit_em(self, X: jnp.ndarray, num_iters: int = 100) -> ModelParams:
        """
        Fit the model using EM.
        """
        params = self.init_params(X)

        for i in range(num_iters):
            # E-step
            qZ = self.E_step(params, X)
            # M-step
            params = self.M_step(qZ, X)

        return params

    def variational_inference(self, 
                              X: jnp.ndarray, 
                              num_iters: int = 100, 
                              learning_rate: float = 1e-3) -> ModelParams:
        """
        Fit the model using variational inference (VI).
        """
        params = self.init_params(X)
        phi = self.init_variational_params(X)

        @jit
        def elbo_fn(params, phi):
            # For the ELBO, we compute E_q[log p(X,Z) - log q(Z)].
            # Typically, we approximate expectations with Monte Carlo samples.
            Z_samples = self.sample_from_q(phi, num_samples=10)
            log_joint_vals = jnp.mean(jax.vmap(lambda z: self.log_joint(params, X, z))(Z_samples))
            log_q_vals = jnp.mean(jax.vmap(lambda z: self.log_q(phi, z))(Z_samples))
            return log_joint_vals - log_q_vals

        elbo_grad = value_and_grad(elbo_fn, argnums=(0,1))

        def update(params, phi):
            elbo_val, (grad_params, grad_phi) = elbo_grad(params, phi)
            params = jax.tree_util.tree_map(lambda p, g: p + learning_rate * g, params, grad_params)
            phi = jax.tree_util.tree_map(lambda p, g: p + learning_rate * g, phi, grad_phi)
            return params, phi, elbo_val

        for iteration in range(num_iters):
            params, phi, elbo_val = update(params, phi)
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, ELBO: {elbo_val}")

        return params
