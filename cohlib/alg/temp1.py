def computer_fisher_info_delta_log_poisson(self, v_ests):
    Cs = [obj.num_neurons for obj in self.trial_objs]
    alphas = [obj.params['alpha'] for obj in self.trial_objs]
    J = self.num_J_vars
    fs = 1000
    delta = 1/fs

    WFkWs = []
    for k in range(self.K):
        v_ests_k = v_ests[k*J:k*J+J]

        x = self.W @ v_ests_k
        lamb = np.exp(alphas[k] + x)
        F_k = np.diag(lamb*delta)

        WFkW = Cs[k] * self.W.T @ F_k @ self.W
        WFkWs.append(WFkW)
    Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

    return -Hessian