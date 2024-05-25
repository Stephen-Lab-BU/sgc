import scipy.optimize as op
from scipy.linalg import block_diag
import numpy as np
from abc import ABC, abstractmethod
mvn = np.random.multivariate_normal

class TrialData():
    def __init__(self, trial_objs, Gamma_inv_prev, W, params, obs_model):
        self.trial_objs = trial_objs
        self.W = W
        self.num_J_vars = W.shape[1]
        self.Gamma_inv_prev = Gamma_inv_prev
        self.K = len(trial_objs)
        self.params = params
        self.obs_model = obs_model

        if self.obs_model == 'poisson':
            self.compute_fisher_info = self.compute_fisher_info_pois
        elif self.obs_model == 'poisson-relu':
            self.compute_fisher_info = self.compute_fisher_info_pois_relu
        elif self.obs_model == 'poisson-id':
            self.compute_fisher_info = self.compute_fisher_info_pois_id
        elif self.obs_model == 'bernoulli':
            self.compute_fisher_info = self.compute_fisher_info_bernoulli
        else:
            raise NotImplementedError

    def cost_func(self):
        def func(v):
            Jv = self.num_J_vars
            group_terms = np.array(
                [obj.cost_func(v[k*Jv:k*Jv + Jv], self.W) 
                for k, obj in enumerate(self.trial_objs)])
            group_term = group_terms.sum()
            # prior_term =  v.T @ self.Gamma_inv_prev @ v 
            prior_term = (1/2) * (v.T @ self.Gamma_inv_prev @ v)

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
            return -grad
        return func

    def cost_hess(self):
        def func(v):
            return self.compute_fisher_info(v)
        return func

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
        Ups_inv = self.compute_fisher_info(mu)
        return mu, Ups_inv


    # TODO - move to SpikeTrial object
    def compute_fisher_info_pois(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        alphas = [obj.params['alpha'] for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):
            v_ests_k = v_ests[k*J:k*J+J]
            # inner = -self.W @ v_ests_k
            # F_k1 = np.diag(np.exp(inner) / (1+ np.exp(inner))**2)

            x = self.W @ v_ests_k
            # lamb = 1/(1 + np.exp(-(self.alphas[k] + x)))
            # F_k = np.diag(lamb*(1-lamb))
            lamb = np.exp(alphas[k] + x)
            F_k = np.diag(lamb)

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return -Hessian

    def compute_fisher_info_pois_relu(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        alphas = [obj.params['alpha'] for obj in self.trial_objs]
        data_processed = [obj.data_processed for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):
            v_ests_k = v_ests[k*J:k*J+J]
            # inner = -self.W @ v_ests_k
            # F_k1 = np.diag(np.exp(inner) / (1+ np.exp(inner))**2)

            x = self.W @ v_ests_k
            # lamb = 1/(1 + np.exp(-(self.alphas[k] + x)))
            # F_k = np.diag(lamb*(1-lamb))
            lamb = alphas[k] + x
            lamb[lamb<=0] = np.nan
            
            F_k = np.diag(np.nan_to_num(data_processed[k] / lamb**2))

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return -Hessian

    def compute_fisher_info_pois_id(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        alphas = [obj.params['alpha'] for obj in self.trial_objs]
        data_processed = [obj.data_processed for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):
            v_ests_k = v_ests[k*J:k*J+J]

            x = self.W @ v_ests_k
            lamb = alphas[k] + x
            
            F_k = np.diag(data_processed[k] / lamb**2)

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return -Hessian

    def compute_fisher_info_bernoulli(self, v_ests):
        Cs = [obj.num_neurons for obj in self.trial_objs]
        alphas = [obj.params['alpha'] for obj in self.trial_objs]
        J = self.num_J_vars

        WFkWs = []
        for k in range(self.K):

            v_ests_k = v_ests[k*J:k*J+J]

            x = self.W @ v_ests_k
            lamb = 1/(1 + np.exp(-(alphas[k] + x)))
            F_k = np.diag(lamb*(1-lamb))

            WFkW = Cs[k] * self.W.T @ F_k @ self.W
            WFkWs.append(WFkW)
        Hessian = -block_diag(*WFkWs) - self.Gamma_inv_prev

        return -Hessian

class SpikeTrial(ABC):
    def __init__(self, data, params, taper=None):
        self.data = data
        self.num_neurons = data.shape[0]
        self.window_length = data.shape[1]
        self.data_type = 'spiking'
        self.data_processed = None
        self.params = params

        if taper is None:
            data_avg = data.astype(int).mean(0)
            self.data_processed = data_avg 

        else:
            data_avg = self.data.astype(int).mean(0)
            # data_avg = data_avg - data_avg.mean()
            self.data_processed = self.taper_data(data_avg, taper)

    def taper_data(self, data_avg, taper):
        x_est = np.log(1/( 1/data_avg-1) );
        tapered = data_avg.copy()
        T = data_avg.size

        for t in range(T):
            if data_avg[t] != 0 and data_avg[t] != 1:
                tapered[t] = 1/(1 + np.exp( -1* x_est[t] * taper[t]))

        return tapered

    @abstractmethod
    def cost_func(self, v, W):
        pass

    @abstractmethod
    def cost_grad(self, v, W):
        pass

class SpikeTrialPoisson(SpikeTrial):
    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v
        lamb_pre = alpha + x
        cost_pre = data * lamb_pre - np.exp(lamb_pre)
        cost = C*cost_pre.sum()

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v

        lamb = np.exp(alpha + x)
        diff = data - lamb
        g_pre = C*np.inner(W.T, diff)
        g = g_pre 

        return g

class SpikeTrialPoissonID(SpikeTrial):
    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v
        lamb = alpha + x

        log_lamb = np.log(lamb)
        cost_pre = data * log_lamb - lamb
        cost = C*cost_pre.sum()

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v

        lamb = alpha + x
        div = data/lamb
        diff = div - np.ones_like(lamb)
        g_pre = C*np.inner(W.T, diff)
        g = g_pre 

        return g


class SpikeTrialPoissonReLU(SpikeTrial):
    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v
        lamb = alpha + x

        lamb[lamb<=0] = np.nan
        
        log_lamb = np.nan_to_num(np.log(lamb))
        cost_pre = data * log_lamb - lamb
        cost = C*cost_pre.sum()

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v

        lamb = alpha + x
        lamb[lamb<=0] = np.nan
        div = np.nan_to_num(data/lamb)
        diff = div - np.ones_like(lamb)
        g_pre = C*np.inner(W.T, diff)
        g = g_pre 

        return g

class SpikeTrialBernoulli(SpikeTrial):
    def cost_func(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v

        lamb_pre = alpha + x
        cost_pre = data * lamb_pre - np.log(1 + np.exp(lamb_pre))
        cost = C*cost_pre.sum()

        return cost

    def cost_grad(self, v, W):
        data = self.data_processed
        C = self.num_neurons
        alpha = self.params['alpha']

        x = W @ v

        lamb = 1/(1+np.exp(-(alpha + x)))
        diff = data - lamb
        g_pre = C*np.inner(W.T, diff)
        g = g_pre 

        return g
