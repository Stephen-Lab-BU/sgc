import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
from jaxopt import ScipyMinimize
from functools import partial

from jaxopt import ScipyBoundedMinimize
# from jaxopt import BoxConstraints

def complex_to_real_matrix(X_c):
    """Convert a complex matrix (K x R) to real matrix (K x 2R) for optimization"""
    return jnp.concatenate([jnp.real(X_c), jnp.imag(X_c)], axis=-1)

def real_to_complex_matrix(X_r):
    """Convert a real matrix (K x 2R) to complex matrix (K x R)"""
    R = X_r.shape[1] // 2
    return X_r[:, :R] + 1j * X_r[:, R:]

def make_loss_fn(S, L):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params
        log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e7))
        sigma2 = jnp.exp(log_sigma2_clipped)

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        # frobenius_penalty = jnp.sum(jnp.square(jnp.abs(B)))
        # lambda_B = 1e-6

        # return L * jnp.real(logdet) + jnp.real(trace_term)  + lambda_B * frobenius_penalty # Ensure output is real scalar
        return L * jnp.real(logdet) + jnp.real(trace_term)  # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factor(S, rank, L, init_sigma2=1e-2, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    # bounds = BoxConstraints(
    #     lb=jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-6)),
    #     ub=jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e6))
    # )
    # solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B", bounds=bounds)
    lb = jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-3)),
    ub = jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e3)),
    bounds = (lb, ub)
    solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B")
    result = solver.run(flat_init, bounds=bounds)
    # solver = ScipyMinimize(fun=flattened_loss, method="BFGS")

    # result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factor(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    return result['Gamma'][None,:,:]

def make_loss_fn_igprior(S, L):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params

        # log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e3))
        # log_sigma2 = log_sigma2_clipped

        sigma2 = jnp.exp(log_sigma2)
        # sigma2 = jnp.exp(log_sigma2)
        # print(f'sigma2: {sigma2}')

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        # Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        # for synthetic data
        # a = 2
        # applied data topK
        # a = 250
        # applied data bottomK

        a = 150
        b = 1e4

        # for synthetic data
        # a = 2
        # b = 1e4
 
        log_prior = -(a + 1) * log_sigma2 - b / sigma2

        return L * jnp.real(logdet) + jnp.real(trace_term) - log_prior # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factor_igprior(S, rank, L, init_sigma2=1e3, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn_igprior(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    solver = ScipyMinimize(fun=flattened_loss, method="BFGS")
    result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factor_igprior(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor_igprior(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    print(f"B shape: {result['B'].shape}")
    return result['Gamma'][None,:,:]


def make_loss_fn_igprior_anucpen(S, L):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params

        # log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e3))
        # log_sigma2 = log_sigma2_clipped

        sigma2 = jnp.exp(log_sigma2)
        # sigma2 = jnp.exp(log_sigma2)
        # print(f'sigma2: {sigma2}')

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        # Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        a = 1
        b = 1e4
        log_prior = -(a + 1) * log_sigma2 - b / sigma2
        # log_prior = 0
        # mu = 1e3
        # tau = 1
        lambda_B = 5e-6
        epsilon=1e-6
        # penalty = lambda_B * smoothed_nuclear_norm(Gamma, epsilon)
        penalty = lambda_B * nuclear_norm(Gamma)
        # log_prior = -0.5 * ((log_sigma2 - mu) / tau)**2
        # # frobenius_penalty = jnp.sum(jnp.square(jnp.abs(B)))
        # lambda_B = 1e-6

        # return L * jnp.real(logdet) + jnp.real(trace_term)  + lambda_B * frobenius_penalty # Ensure output is real scalar
        return L * jnp.real(logdet) + jnp.real(trace_term) + penalty - log_prior # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factor_igprior_anucpen(S, rank, L, init_sigma2=1e3, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn_igprior_anucpen(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    solver = ScipyMinimize(fun=flattened_loss, method="BFGS")
    result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factor_igprior_anucpen(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor_igprior_anucpen(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    return result['Gamma'][None,:,:]

def make_loss_fn_mk(S, L):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params
        log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e7))
        sigma2 = jnp.exp(log_sigma2_clipped)

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        BtB = B.conj().T @ B
        prior_eigs = jnp.linalg.eigvalsh(BtB)
        logdet_prior = jnp.sum(jnp.log(jnp.real(prior_eigs)))

        # frobenius_penalty = jnp.sum(jnp.square(jnp.abs(B)))
        # lambda_B = 1e-6

        # return L * jnp.real(logdet) + jnp.real(trace_term)  + lambda_B * frobenius_penalty # Ensure output is real scalar
        return L * jnp.real(logdet) + jnp.real(trace_term) + 0.5*(K - R - 1) * logdet_prior # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factormk(S, rank, L, init_sigma2=1e-2, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn_mk(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    # bounds = BoxConstraints(
    #     lb=jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-6)),
    #     ub=jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e6))
    # )
    # solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B", bounds=bounds)
    lb = jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-3)),
    ub = jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e3)),
    bounds = (lb, ub)
    solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B")
    result = solver.run(flat_init, bounds=bounds)
    # solver = ScipyMinimize(fun=flattened_loss, method="BFGS")

    # result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factormk(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    return result['Gamma'][None,:,:]

def make_loss_fn_frobpen(S, L):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params
        log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e7))
        sigma2 = jnp.exp(log_sigma2_clipped)

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        lambda_B = 1e-4
        penalty = lambda_B * jnp.sum(jnp.square(jnp.abs(B))) 

        # frobenius_penalty = jnp.sum(jnp.square(jnp.abs(B)))
        # lambda_B = 1e-6

        # return L * jnp.real(logdet) + jnp.real(trace_term)  + lambda_B * frobenius_penalty # Ensure output is real scalar
        return L * jnp.real(logdet) + jnp.real(trace_term) + penalty # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factor_frobpen(S, rank, L, init_sigma2=1e-2, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn_frobpen(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    # bounds = BoxConstraints(
    #     lb=jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-6)),
    #     ub=jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e6))
    # )
    # solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B", bounds=bounds)
    lb = jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-3)),
    ub = jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e3)),
    bounds = (lb, ub)
    solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B")
    result = solver.run(flat_init, bounds=bounds)
    # solver = ScipyMinimize(fun=flattened_loss, method="BFGS")

    # result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factor_frobpen(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor_frobpen(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    return result['Gamma'][None,:,:]

def smoothed_nuclear_norm(Gamma, epsilon):
    """Smoothed surrogate for nuclear norm: sum sqrt(lambda^2 + epsilon)"""
    eigvals = jnp.linalg.eigvalsh(Gamma)
    return jnp.sum(jnp.sqrt(eigvals**2 + epsilon))

def nuclear_norm(Gamma):
    """Smoothed surrogate for nuclear norm: sum sqrt(lambda^2 + epsilon)"""
    eigvals = jnp.linalg.eigvalsh(Gamma)
    return jnp.sum(eigvals)

def make_loss_fn_anucpen(S, L, epsilon=1e-2):
    """Returns a function of real-valued B (K x 2R) and scalar log_sigma2"""
    def loss_fn(params):
        B_r, log_sigma2 = params
        log_sigma2_clipped = jnp.clip(log_sigma2, jnp.log(1e-3), jnp.log(1e7))
        sigma2 = jnp.exp(log_sigma2_clipped)

        # sigma2 = jnp.exp(log_sigma2)
        B = real_to_complex_matrix(B_r)
        K = S.shape[0]

        # Gamma = BB^H + sigma^2 * I
        BBH = B @ B.conj().T
        Gamma = BBH + sigma2 * jnp.eye(K)
        # Gamma += 1e-3 * jnp.eye(K)

        # Cholesky for log determinant
        L_chol = jnp.linalg.cholesky(Gamma)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diag(L_chol))))  # Make real

        # Solve Gamma^{-1} @ S
        Gamma_inv_S = cho_solve((L_chol, True), S)
        trace_term = jnp.trace(Gamma_inv_S)

        lambda_B = 1e-6
        penalty = lambda_B * smoothed_nuclear_norm(Gamma, epsilon)

        # frobenius_penalty = jnp.sum(jnp.square(jnp.abs(B)))
        # lambda_B = 1e-6

        # return L * jnp.real(logdet) + jnp.real(trace_term)  + lambda_B * frobenius_penalty # Ensure output is real scalar
        return L * jnp.real(logdet) + jnp.real(trace_term) + penalty # Ensure output is real scalar
    return loss_fn

def optimize_m_step_factor_anucpen(S, rank, L, init_sigma2=1e-2, key=None):
    """
    Perform M-step to optimize B and sigma^2 given S = E[z z^H].
    S: (K, K) Hermitian matrix
    rank: desired low-rank (R)
    L: number of trials or expected observations
    """
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize B randomly
    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)
    log_sigma2_init = jnp.log(init_sigma2)
    print(log_sigma2_init)

    loss_fn = make_loss_fn_anucpen(S, L)

    # Combine parameters into tuple
    params_init = (B_init_r, log_sigma2_init)

    def flattened_loss(flat_params):
        B_r_flat, log_sigma2 = flat_params[:-1], flat_params[-1]
        B_r = B_r_flat.reshape((K, 2 * rank))
        return loss_fn((B_r, log_sigma2))

    # Flatten for JAXopt
    flat_init = jnp.concatenate([params_init[0].ravel(), jnp.array([params_init[1]])])

    # bounds = BoxConstraints(
    #     lb=jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-6)),
    #     ub=jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e6))
    # )
    # solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B", bounds=bounds)
    lb = jnp.full(flat_init.shape, -jnp.inf).at[-1].set(jnp.log(1e-3)),
    ub = jnp.full(flat_init.shape, jnp.inf).at[-1].set(jnp.log(1e3)),
    bounds = (lb, ub)
    solver = ScipyBoundedMinimize(fun=flattened_loss, method="L-BFGS-B")
    result = solver.run(flat_init, bounds=bounds)
    # solver = ScipyMinimize(fun=flattened_loss, method="BFGS")

    # result = solver.run(flat_init)
    flat_opt = result.params

    # Unpack result
    B_r_opt = flat_opt[:-1].reshape((K, 2 * rank))
    log_sigma2_opt = flat_opt[-1]
    sigma2_opt = jnp.exp(log_sigma2_opt)
    B_opt = real_to_complex_matrix(B_r_opt)

    # Final Gamma estimate
    Gamma_opt = B_opt @ B_opt.conj().T + sigma2_opt * jnp.eye(K)

    return {
        "B": B_opt,
        "sigma2": sigma2_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }

def m_step_factor_anucpen(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_factor_anucpen(S, rank, L)
    print(f"sigma2: {result['sigma2']}")
    return result['Gamma'][None,:,:]


def make_loss_fn_strict_lowrank(S, L):
    def loss_fn(B_r):
        B = real_to_complex_matrix(B_r)
        K, R = B.shape

        # Compute Gamma = BB^H and its pseudo-inverse using Woodbury
        BBH = B @ B.conj().T
        BtB = B.conj().T @ B
        BtB_inv = jnp.linalg.inv(BtB)
        Gamma_inv = B @ BtB_inv @ B.conj().T

        # Gamma_inv = jnp.linalg.pinv(BBH)

        # Compute log determinant of BB^H
        eigvals = jnp.linalg.eigvalsh(BBH)
        # Retain only non-zero eigvals (numerically)
        eigvals = jnp.where(eigvals > 1e-10, eigvals, 1e-10)
        logdet = jnp.sum(jnp.log(jnp.real(eigvals)))

        trace_term = jnp.trace(Gamma_inv @ S)

        return L * (jnp.real(logdet)) + jnp.real(trace_term)
    return loss_fn

def optimize_m_step_strict_lowrank(S, rank, L, key=None):
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)

    loss_fn = make_loss_fn_strict_lowrank(S, L)
    solver = ScipyMinimize(fun=loss_fn, method="BFGS")
    result = solver.run(B_init_r)

    B_r_opt = result.params
    B_opt = real_to_complex_matrix(B_r_opt)
    Gamma_opt = B_opt @ B_opt.conj().T

    return {
        "B": B_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }


def m_step_strictlr(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_strict_lowrank(S, rank, L)
    return result['Gamma'][None,:,:]


def make_loss_fn_strict_lowrank_mk(S, L):
    def loss_fn(B_r):
        B = real_to_complex_matrix(B_r)
        K, R = B.shape

        # Compute Gamma = BB^H and its pseudo-inverse using Woodbury
        BBH = B @ B.conj().T
        BtB = B.conj().T @ B
        BtB_inv = jnp.linalg.inv(BtB)
        Gamma_inv = B @ BtB_inv @ B.conj().T

        # Gamma_inv = jnp.linalg.pinv(BBH)

        # Compute log determinant of BB^H
        eigvals = jnp.linalg.eigvalsh(BBH)
        # Retain only non-zero eigvals (numerically)
        eigvals = jnp.where(eigvals > 1e-10, eigvals, 1e-10)
        logdet = jnp.sum(jnp.log(jnp.real(eigvals)))

        trace_term = jnp.trace(Gamma_inv @ S)

        prior_eigs = jnp.linalg.eigvalsh(BtB)
        logdet_prior = jnp.sum(jnp.log(jnp.real(prior_eigs)))

        return L * (jnp.real(logdet)) + jnp.real(trace_term) + 0.5*(K - R - 1) * logdet_prior
    return loss_fn

def optimize_m_step_strict_lowrank_mk(S, rank, L, key=None):
    K = S.shape[0]
    if key is None:
        key = jax.random.PRNGKey(0)

    B_init = jax.random.normal(key, (K, rank)) + 1j * jax.random.normal(key, (K, rank))
    B_init_r = complex_to_real_matrix(B_init)

    loss_fn = make_loss_fn_strict_lowrank_mk(S, L)
    solver = ScipyMinimize(fun=loss_fn, method="BFGS")
    result = solver.run(B_init_r)

    B_r_opt = result.params
    B_opt = real_to_complex_matrix(B_r_opt)
    Gamma_opt = B_opt @ B_opt.conj().T

    return {
        "B": B_opt,
        "Gamma": Gamma_opt,
        "loss": result.state.fun_val
    }# Next let's try no Sigma

def m_step_strictlrmk(alphas_outer, Upss, rank):
    S = (alphas_outer + Upss).sum(-1).squeeze()
    L = alphas_outer.shape[-1]
    result = optimize_m_step_strict_lowrank_mk(S, rank, L)
    return result['Gamma'][None,:,:]