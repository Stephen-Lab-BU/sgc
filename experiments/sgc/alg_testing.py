import numpy as np
from cohlib.utils import pickle_open
from cohlib.alg.laplace_sgc import SpikeTrial, TrialData

from cohlib.utils import conv_complex_to_real

def conv_zs_to_vs(zs, dc=False):
    L = zs.shape[0]
    J = zs.shape[1]
    if dc:
        vs = np.zeros((L, int(J*2))+1)
        vs[:,0] = zs[:,zs[0].real]
        dcshift = 1
    else:
        vs = np.zeros((L, int(J*2)))
        dcshift = 0

    # temp = []
    for l in range(L):
        for j in range(int(J)):
            ind = j + dcshift
            z = zs[l,ind]
            v1,v2 = conv_complex_to_real(z)

            vi = j*2 + dcshift
            vs[l,vi] = v1
            vs[l,vi+1] = v2
            # temp.append(v1)
            # temp.append(v2)

    # return vs, temp
    return vs
L = 25
sample_length = 1000
C = 30
seed = 8
mu = -3.5

save_path = f'saved/synthetic_data/simple_synthetic_{L}_{sample_length}_{C}_{mu}_{seed}'

data_load = pickle_open(save_path)

pp_params = data_load['observed']['pp_params']
spikes1 = data_load['observed']['spikes'][0]
spikes2 = data_load['observed']['spikes'][1]

xns = data_load['latent']['xns']
xys = data_load['latent']['xys']

xnfs = data_load['latent']['xnfs']
xyfs = data_load['latent']['xyfs']
Gamma = data_load['latent']['Gamma']
true_coh = data_load['meta']['coh_true']
coh1 = data_load['meta']['coh_direct_est']

freqs = data_load['meta']['freqs']
spikes1.shape
spikes1.shape

spikes = [spikes1, spikes2]
from cohlib.alg.transform import generate_harmonic_dict
sample_length = data_load['meta']['sample_length']
fs = data_load['meta']['fs']
frange = [0, int(fs/2)]
res = fs/sample_length

W = generate_harmonic_dict(sample_length, fs, res, frange)


from cohlib.alg.laplace_sgc import TrialData
# trial_obj = TrialData(trial_data, gamma_prev_inv, W)
trial = 0
K = 2
trial_spike_objs = []
trial_data = []
for k in range(K):
    data_kl = spikes[k][trial,:,:]

    spk_kl = SpikeTrial(data_kl)

    trial_spike_objs.append(spk_kl)
    trial_data.append(data_kl)

trial_z1 = conv_zs_to_vs(xnfs[0:1,:]).squeeze()
trial_z2 = conv_zs_to_vs(xyfs[0:1,:]).squeeze()
trial_z = np.concatenate([trial_z1, trial_z2])
# test = trial_data[0].cost_func(trial_z, W)
trial_z.shape

num_freqs = W.shape[1]
num_freqs

gamma_prev_inv = np.eye(num_freqs*K) + np.random.randn(num_freqs*K, num_freqs*K)*0.1
gamma_prev_inv.shape
gamma_init = np.eye(num_freqs*K)*gamma_prev_inv

# trial_obj = TrialData(trial_spike_objs, np.eye(num_freqs*K)*gamma_prev_inv, W)
# test = trial_obj.cost_grad()
# test(trial_z)
# # spike_trial_obj.cost_func(trial_z1, W)
# nf = W.shape[1]
# group_terms = np.array(
#     [obj.cost_grad(trial_z[k*nf:k*nf + nf], W) 
#     for k, obj in enumerate(trial_spike_objs)])
# group_terms
# group_terms.flatten()
# testMu, testH = trial_obj.laplace_approx()


def get_trial_obj(data, l, W, gamma_prev):
    trial_data = [group_data[l,:,:] for group_data in data]
    spike_objs = [SpikeTrial(data) for data in trial_data]
    trial_obj = TrialData(spike_objs, gamma_prev, W)
    return trial_obj

spikes_short = [spikes[i][:5,:,:] for i in range(len(spikes))]

test = get_trial_obj(spikes_short, 0, W, np.eye(num_freqs*K)*gamma_prev_inv)
# test.laplace_approx()
# fit_sgc_model(data, W, inits, num_em_iters=10, max_approx_iters=10, track=False)

from cohlib.alg.em_sgc import fit_sgc_model
inits = {'Gamma': gamma_init}

test = fit_sgc_model(spikes_short, W, inits, num_em_iters=2, max_approx_iters=10, track=False)

