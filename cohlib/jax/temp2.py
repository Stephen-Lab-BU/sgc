class OldOptimMod():
    def __init__(self, data, gamma_inv, params, obs_type, track=False):
        self.data = data
        self.gamma_inv = gamma_inv
        self.params = params
        self.track = track

        self.obs_var = params['obs_var']
        self.Wv = params['Wv']
        self.num_J_vars = self.Wv.shape[1]
        self.K = data.shape[1]

        # print(f'Confirming decon-mod: {self.params["decon_mod"]}')

        if obs_type == 'gaussian':
            pass
        else:
            raise NotImplementedError

        nz = params['nonzero_inds']
        sample_length = self.Wv.shape[0]

        invQ = jnp.diag(jnp.ones(sample_length)*(1/self.obs_var))

        obs_objs = [GaussianTrial(data[None,:,i], invQ) for i in range(self.K)] 
        # gamma_inv_oldformat = construct_Gamma_full_real(self.gamma_inv[nz,:,:], 
                                #  self.K, self.num_J_vars, invert=False)
        gamma_inv_oldformat = 4*construct_Gamma_full_real_mod(self.gamma_inv[nz,:,:], 
                                self.K, self.num_J_vars, invert=False)
        trial_obj = TrialDataGaussian(obs_objs, gamma_inv_oldformat, self.Wv)

        self.cost_func = trial_obj.cost_func()
        self.cost_grad = trial_obj.cost_grad()
        self.cost_hess = trial_obj.cost_hess()

        self.optim_result = None

    def eval_cost(self, zs):
        vs = conv_z_to_v(zs, axis=0)
        vs_flat = jnp.concatenate([vs[:,k] for k in range(self.K)])

        cost_real = self.cost_func(vs_flat)
        grad_real = self.cost_grad(vs_flat)
        hess_full_real = self.cost_hess(vs_flat)

        grad = conv_grad_old_r2c(grad_real, self.K)
        if self.params['decon_mod'] is False:
            hess = deconstruct_Gamma_full_real(hess_full_real, self.K, self.num_J_vars)
        else: 
            hess = deconstruct_Gamma_full_real_mod(hess_full_real, self.K, self.num_J_vars)

        return cost_real, grad, hess
        # return cost_real, grad, hess, hess_full_real

    def run_e_step(self, zs_init, num_iters):
        zs_est = zs_init
        if self.track is True:
            track_zs = [zs_init]
            track_cost = []
            track_grad = []
            track_hess = []

        for _ in range(num_iters):
            cost, grad, hess = self.eval_cost(zs_est)
            # cost, grad, hess, hess_real = self.eval_cost(zs_est)
            hess_inv = jnp.linalg.inv(hess)

            zs_est = zs_est - jnp.einsum('nki,ni->nk', hess_inv, grad)
            if self.track is True:
                track_zs.append(zs_est)
                track_cost.append(cost)
                track_grad.append(grad) 
                track_hess.append(hess) 


        if self.track is True:
            result = OptimResult(zs_est, hess, track_zs, track_cost, track_grad, track_hess)
        else:
            result = OptimResult(zs_est, hess)
        self.result = result
        # vs_est = conv_mu_old_r2c(zs_est, self.K)
        # self.real_result = OptimResultReal(vs_est, hess_real)
