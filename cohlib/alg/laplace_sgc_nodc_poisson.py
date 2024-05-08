import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
mvn = np.random.multivariate_normal

class TrialDataPoisson():
    def __init__(self, trial_objs, Gamma_inv_prev, W, alphas):
        self.trial_objs = trial_objs
        self.W = W
        self.num_J_vars = W.shape[1]
        self.Gamma_inv_prev = Gamma_inv_prev
        self.K = len(trial_objs)
        self.alphas = alphas

    def cost_func(self):
        # d = self.Gamma_inv_prev.shape[0]
        # prior_term_const = np.log(np.linalg.det(np.linalg.inv(self.Gamma_inv_prev))) + d*np.log(2*np.pi)
        prior_term_const = 0
        def func(v):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_func(v[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.sum()
            # prior_term =  v.T @ self.Gamma_inv_prev @ v 
            prior_term = (1/2) * (prior_term_const + v.T @ self.Gamma_inv_prev @ v)

            cost = group_term - prior_term
            # print(-cost)
            return -cost
        return func
    
    def cost_grad(self):
        def func(v):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_grad(v[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.flatten()
            prior_term = self.Gamma_inv_prev @ v

            grad = group_term - prior_term
            # NOTE fixing DC
            # for k in range(self.K):
            #     grad[k*self.num_J_vars] = 0 
            # print(np.abs(grad).mean())
            return -grad
        return func

    def cost_hess(self):
        def func(v):
            Cs = [obj.num_neurons for obj in self.trial_objs]
            J = self.num_J_vars

            WFkWs = []
            for k in range(self.K):
                v_ests_k = v[k*J:k*J+J]
                # inner = -self.W @ v_ests_k
                # F_k = np.diag(np.exp(inner) / np.exp(inner)**2)

                x = self.W @ v_ests_k
                # lamb = 1/(1 + np.exp(-(self.alphas[k] + x)))
                # F_k = np.diag(lamb*(1-lamb))
                lamb = np.exp(self.alphas[k] + x)
                F_k = np.diag(lamb)

                WFkW = Cs[k] * self.W.T @ F_k @ self.W
                WFkWs.append(WFkW)
            # Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev
            hess = -block_diag(*WFkWs) - self.Gamma_inv_prev


            return -hess
        return func
        # return function of v 

    def laplace_approx(self, max_iter=100):

        cost_func_optim = self.cost_func()
        cost_grad_optim = self.cost_grad()
        cost_hess_optim = self.cost_hess()

        init = np.zeros(self.num_J_vars*self.K)

        # NOTE init at 0 gives 0.5 probability at each time point
        Result = op.minimize(fun=cost_func_optim, x0=init,
                        # jac=cost_grad_optim, method='BFGS', 
                        jac=cost_grad_optim, hess=cost_hess_optim, method='Newton-CG', 
                        options={'maxiter':max_iter, 'disp':False})

        mu = Result.x
        # return mu
        Ups_inv = self.compute_Hessian(mu)
        return mu, Ups_inv


    def compute_Hessian(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):
            v_ests_k = v_ests[k*J:k*J+J]
            # inner = -self.W @ v_ests_k
            # F_k1 = np.diag(np.exp(inner) / (1+ np.exp(inner))**2)

            x = self.W @ v_ests_k
            # lamb = 1/(1 + np.exp(-(self.alphas[k] + x)))
            # F_k = np.diag(lamb*(1-lamb))
            lamb = np.exp(self.alphas[k] + x)
            F_k = np.diag(lamb)

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return Hessian


# TODO make ABC for data models
class SpikeTrialPoisson():
    def __init__(self, data, alpha, taper=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'spiking'
        self.data_processed = None
        self.alpha = alpha


        if taper is None:
            data_avg = data.astype(int).mean(0)
            # self.data_processed = data_avg - data_avg.mean()
            self.data_processed = data_avg 

        else:
            data_avg = self.data.astype(int).mean(0)
            data_avg = data_avg - data_avg.mean()
            self.data_processed = self.taper_data(data_avg, taper)

    def taper_data(self, data_avg, taper):
        x_est = np.log(1/( 1/data_avg-1) );
        tapered = data_avg.copy()
        T = data_avg.size

        for t in range(T):
            if data_avg[t] != 0 and data_avg[t] != 1:
                tapered[t] = 1/(1 + np.exp( -1* x_est[t] * taper[t]))

        return tapered

    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons

        x = W @ v
        # lamb_pre = mus[:,None] + x
        # cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
        # cost = cost_pre.sum()

        lamb_pre2 = self.alpha + x
        # cost_pre2 = data * lamb_pre2 - np.log(1 + np.exp(lamb_pre2))
        cost_pre2 = data * lamb_pre2 - np.exp(lamb_pre2)
        cost2 = C*cost_pre2.sum()

        return cost2

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons

        x = W @ v

        # lamb = 1/(1+np.exp(-(mus[:,None] + x)))
        # lamb = 1/(1+np.exp(-(self.alpha + x)))
        lamb = np.exp(self.alpha + x)
        diff = data - lamb
        g_pre = C*np.inner(W.T, diff)
        # g_pre = (np.inner(W.T, diff)).sum(1)
        g = g_pre 

        return g
