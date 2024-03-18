import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
from cohlib.utils import get_dcval
mvn = np.random.multivariate_normal

# TODO rename z -> v

class TrialData():
    def __init__(self, trial_objs, Gamma_inv_prev, W, prev_est):
        self.trial_objs = trial_objs
        self.W = W
        self.num_J_vars = W.shape[1]
        self.Gamma_inv_prev = Gamma_inv_prev
        self.K = len(trial_objs)
        self.prev_est = prev_est

    def cost_func(self):
        d = self.Gamma_inv_prev.shape[0]
        prior_term_const = np.log(np.linalg.det(np.linalg.inv(self.Gamma_inv_prev))) + d*np.log(2*np.pi)
        # prior_term_const = 0
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
                lamb = 1/(1 + np.exp(-x))
                F_k = np.diag(lamb*(1-lamb))

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
        # NOTE initing with prev_est
        # init = self.prev_est
        # NOTE fixing dc
        # J = int(self.W.shape[0] / 2)
        # for k in range(self.K):
        #     k_mu = self.trial_objs[k].mu
        #     k_dc = get_dcval(k_mu, J, 'real')
        #     init[k*self.num_J_vars] = k_dc

        # NOTE initialize means at true values
        # J = int(self.W.shape[0] / 2)
        # for k in range(self.K):
        #     loc = k*self.num_J_vars
        #     k_mu = self.trial_objs[k].mu
        #     k_dc = get_dcval(k_mu, J, 'real')
        #     init[k*self.num_J_vars] = k_dc

        # NOTE init at 0 gives 0.5 probability at each time point
        # NOTE only on first em iter if initing with prev_est
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
            lamb = 1/(1 + np.exp(-x))
            F_k = np.diag(lamb*(1-lamb))

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return Hessian


# TODO make ABC for data models
class SpikeTrial():
    def __init__(self, data, mu=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'spiking'
        self.mu=mu

    # def taper():
    #     pass

    def cost_func(self, v, W):
        data = self.data.astype(int)
        C = self.num_neurons
        # mus = self.mu * np.ones(C) 
        mus = 0 * np.ones(C) 

        x = W @ v
        lamb_pre = mus[:,None] + x
        cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
        cost = cost_pre.sum()

        lamb_pre2 = x
        cost_pre2 = data.mean(0) * lamb_pre2 - np.log(1 + np.exp(lamb_pre2))
        cost2 = C*cost_pre2.sum()

        return cost

    def cost_grad(self, v, W):
        data = self.data.astype(bool)
        C = self.num_neurons
        # mus = self.mu * np.ones(C)
        mus = 0 * np.ones(C)

        x = W @ v

        lamb = 1/(1+np.exp(-(mus[:,None] + x)))
        # lamb = 1/(1+np.exp(-x))
        diff = data - lamb
        g_pre = (np.inner(W.T, diff)).sum(1)
        # g_pre = (np.inner(W.T, diff)).sum(1)
        g = g_pre 

        return g

    def cif(self, x):
        lamb_pre = x
        return 1 / (1 + np.exp(-lamb_pre))



    def cost_func_orig(self, v, W):
        data = self.data.astype(bool)
        C = self.num_neurons
        # mus = self.mu * np.ones(C) 
        mus = 0 * np.ones(C) 

        x = W @ v
        lamb_pre = mus[:,None] + x
        cost_pre = (data * lamb_pre - np.log(1 + np.exp(lamb_pre)))
        cost = cost_pre.sum() 

        return cost

