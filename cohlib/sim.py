import numpy as np

def draw_spikes(x, mu, beta):
    lamb = 1 / (1 + np.exp(-(mu + beta*x)))
    spikes = np.random.binomial(1, lamb)
    return spikes
                
def spikes_from_xns(x, pp_params, n_trials, T):
    mu = pp_params['mu']
    beta = pp_params['beta']
    C = mu.size
    spikes = np.zeros((n_trials, C, T))
    for l in range(n_trials):
        spikes_l = np.stack([draw_spikes(x[l,:], mu[c], beta[c]) for c in range(C)])
        spikes[l,:,:] = spikes_l
    return spikes