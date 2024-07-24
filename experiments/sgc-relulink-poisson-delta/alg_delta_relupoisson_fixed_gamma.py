import argparse

import numpy as np

from cohlib.alg.transform import construct_real_idft_mod

from cohlib.utils import pickle_save, pickle_open
from funcs.deltarelu import load_and_fit_model


# variables we would like to be able to set through CLI:
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_length', type=int, default=500)
    parser.add_argument('L', type=int, default=30)
    parser.add_argument('K', type=int, default=2)
    parser.add_argument('C', type=int, default=25)
    parser.add_argument('alpha', nargs='?', type=float, default=-3.5)
    parser.add_argument('num_em', nargs='?', type=int, default=10)
    parser.add_argument('init_type', nargs='?', type=str, default='flat')
    parser.add_argument('optim_type', nargs='?', type=str, default='BFGS')
    parser.add_argument('rho', nargs='?', type=int, default=0)
    parser.add_argument('kappa', nargs='?', type=int, default=0)
    parser.add_argument('-seed', nargs='?', type=int, default=7)
    args = parser.parse_args()

    store_spikes = True
    print(f'store_spikes: {store_spikes}')
    sample_length = args.sample_length # sample length (was slen in notebook)
    seed = args.seed
    L = args.L
    C = args.C
    K = args.K
    alpha = args.alpha
    num_em = args.num_em
    init_type = args.init_type
    optim_type = args.optim_type
    fs=1000

    np.random.seed(seed)

    if args.rho == 0:
        rho = None
        kappa = None
        print(f"Fitting poisson data with ReLU link, L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    else: 
        rho = args.rho
        kappa = args.kappa
        print(f"Fitting (regularized) poisson data ReLU link. Params - rho: {rho}, kappa: {kappa}, L: {L}, K: {K}, sample_length: {sample_length}, C: {C}, alpha: {alpha}, seed: {seed}")
    print(f'Using {init_type} init and {optim_type} for optimization.')

    data_path = f'saved/synthetic_data/simple_latent_deltarelu_{K}_{L}_{sample_length}_{seed}.pkl'
    save_path = f'saved/fitted_models/simple_deltarelu_poisson_em{num_em}_{K}_{L}_{sample_length}_{C}_{alpha}_{seed}_{init_type}_{optim_type}_fitted.pkl'

    # data_load = pickle_open(load_path)
    data_load = pickle_open(data_path)
    freqs = data_load['meta']['freqs']

    J_orig = int(sample_length / 2)
    J_new = np.where(freqs > 50)[0][0] - 1
    Wv = construct_real_idft_mod(sample_length, J_orig, J_new, fs)

    save_dict = load_and_fit_model(Wv, data_load, C, alpha, init_type, optim_type, rho, kappa, store_spikes, num_em)
    pickle_save(save_dict, save_path)

if __name__=="__main__":
    run()