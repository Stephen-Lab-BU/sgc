from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
# NOTE This is for oracle estimate - need to write out alternative *latent cost*
# TODO write get_cost_func_e_step that uses low rank latent - refactor get_cost_func_e_step to make more general
# - really should have latent/obs cost func be connected to model object on instantiation... so latent / obs models are abstract class with cost_func etc
# NOTE This is for oracle estimate - need to write out alternative *latent cost*
def cost_func_full(ev, Sigma_ests, u):
    L = Sigma_ests.shape[-1]
    uuH = jnp.outer(u, u.conj())

    logL = 0
    Sigma_ests_projected = jnp.einsum('ij,knl->inl', uuH, Sigma_ests)
    trace_sum = jnp.trace(Sigma_ests_projected).sum()

    logL = -L*jnp.log(ev) - (1/ev)*trace_sum
    return -logL.real

def cost_func_full_eigvec(u, Sigma_ests, ev):
    return cost_func_full(ev, Sigma_ests, u)

def eigvec_optim(ev_est, Sigma_ests, u_init, step_size=1, num_steps=1, method='GD', ts=False):
    cost_func = partial(cost_func_full_eigvec, Sigma_ests=Sigma_ests, ev=ev_est)
    cost_grad = jax.grad(cost_func)
    nu = step_size
    renorm = True
    method = 'GD'

    u_ests = []
    costs = []
    u_est = u_init
    u_ests.append(u_est)
    c = cost_func(u_est)
    costs.append(c)
    for s in range(num_steps):
        g = cost_grad(u_est).conj()
        if method == 'Newton':
            raise NotImplementedError
            # H_real = cost_hess(u_est.real)
            # H_imag = cost_hess(u_est.imag)
            # H = (H_real - H_imag*1j).conj()

            # u_est = u_est - jnp.linalg.inv(H) @ g
        elif method == 'GD':
            u_est = u_est - nu*g
        else: 
            raise NotImplementedError

        if renorm is True:
            u_est = u_est/jnp.linalg.norm(u_est)
            # angle1 = jnp.angle(u_est)[0]
            # u_est = u_est * jnp.exp(-1j*angle1)

        u_ests.append(u_est)

        c = cost_func(u_est)
        costs.append(c)
    if ts is True:
        return u_ests, costs
    else:
        return u_est

def m_step_lowrank_eigvec(alphas_outer, Upss, eigvals, lrccn_prev, params):
    Sigma_ests = (alphas_outer + Upss)
    J = lrccn_prev.Nnz
    K = lrccn_prev.dim

    eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)
    Sigma_ests = (alphas_outer + Upss)

    for j in range(J):
        if params['init_type'] == 'random':
            init_seed = params['m_step_seed']
            u_init = jr.normal(jr.key(j+init_seed), (K,)) + jr.normal(jr.key(j+init_seed+1), (K,))*1j
            u_init = u_init / jnp.linalg.norm(u_init)
        elif params['init_type'] == 'warm_start':
            u_init = lrccn_prev.eigvecs[j,:,0]
        else: 
            raise ValueError

        eigval = eigvals[j,0]
        Sigma_ests_j = Sigma_ests[j,:,:,:]

        u_est = eigvec_optim(eigval, Sigma_ests_j, u_init, ts=False)
        
        # print(f'NOTE: {u_est.shape}')
        eigvecs_update = eigvecs_update.at[j,:,0].set(u_est)
    
    return eigvals, eigvecs_update

def m_step_lowrank_eigval(alphas_outer, Upss, eigvecs, lrccn_prev):
    Sigma_ests = (alphas_outer + Upss)
    eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
    J = lrccn_prev.Nnz

    for j in range(J):
        Sigma_ests_j = Sigma_ests[j,:,:,:]
        u = eigvecs[j,:,:].squeeze()

        uuH = jnp.outer(u, u.conj())

        ev_est = jnp.trace(uuH @ Sigma_ests_j.mean(-1)).real
        
        # print(f'NOTE: {u_est.shape}')
        eigvals_update = eigvals_update.at[j,0].set(ev_est)
    
    return eigvals_update, eigvecs

# TODO clean up m_step options and deprecate unused
def m_step_lowrank_custom(alphas_outer, Upss, params):
    lrccn_prev = params['lrccn_prev']
    rank = lrccn_prev.rank
    ts_flag = params.get('ts_flag')

    fixed_u_mods = ['fixed_u_true', 'fixed_u_oracle']
    fixed_eigval_mods = ['fixed_eigval_true', 'fixed_eigval_oracle']

    if rank != 1:
        raise NotImplementedError


    if ts_flag in fixed_u_mods:
        print(f'M-Step: Estimating eigval; eigvec held using {ts_flag}')
        eigvecs = params['u']
        return m_step_lowrank_eigval(alphas_outer, Upss, eigvecs, lrccn_prev)

    elif ts_flag in fixed_eigval_mods:
        eigvals = params['eigvals']
        print(f'M-Step: Estimating eigvec; eigval held using {ts_flag}')
        return m_step_lowrank_eigvec(alphas_outer, Upss, eigvals, lrccn_prev, params)

    else:

        # Sigma_ests = (alphas_outer + Upss)

        # # if ts_flag == 'mstep_init_prev':
        # #     u_init 
        # u_init = jr.normal(jr.key(init_seed), (K,)) + jr.normal(jr.key(init_seed+1), (K,))*1j
        # u_init = u_init / jnp.linalg.norm(u_init)
        # u_est = u_init
        # # print(f'INIT SHAPE: {u_init.shape}')

        # eigvals_update = jnp.zeros_like(lrccn_prev.eigvals)
        # eigvecs_update = jnp.zeros_like(lrccn_prev.eigvecs)

        # M = 5
        # for j in range(J):
        #     Sigma_ests_j = Sigma_ests[j,:,:,:]
        #     for m in range(M):
        #         uuH = jnp.outer(u_est, u_est.conj())

        #         ev_est = jnp.trace(uuH @ Sigma_ests_j.mean(-1)).real

        #         u_est = eigvec_optim(ev_est, Sigma_ests_j, u_est, ts=False)
            
        #     # print(f'NOTE: {u_est.shape}')
        #     eigvals_update = eigvals_update.at[j,0].set(ev_est)
        #     eigvecs_update = eigvecs_update.at[j,:,0].set(u_est)
        
        # return eigvals_update, eigvecs_update
        
        # TODO update to use m_step_lowrank_eigX functions 
        raise NotImplementedError
