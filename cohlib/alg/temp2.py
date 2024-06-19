def compute_fisher_info_delta_relu_poisson(self, v_ests):
    Cs = [obj.num_neurons for obj in self.trial_objs]
    alphas = [obj.params['alpha'] for obj in self.trial_objs]
    data_processed = [obj.data_processed for obj in self.trial_objs]
    J = self.num_J_vars

    WFkWs = []
    for k in range(self.K):
        v_ests_k = v_ests[k*J:k*J+J]
        x = self.W @ v_ests_k

        lamb = alphas[k] + x
        lamb_relu = np.copy(lamb)
        lamb_relu[lamb_relu<=0] = np.nan
        
        F_k = np.diag(np.nan_to_num(data_processed[k] / lamb_relu**2, nan=0, neginf=0, posinf=0))

        WFkW = Cs[k] * self.W.T @ F_k @ self.W
        WFkWs.append(WFkW)
    Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

    return -Hessian