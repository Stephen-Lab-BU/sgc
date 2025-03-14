from cohlib.conv import *
import numpy as np
import pickle

def gamma_root():
    return '/projectnb/stephenlab/jtauber/cohlib/hydra/gammas'


def get_dcval(mean, J):
    """
    Convert value of mean in log-linear space (alpha) to
    DC term for spectrum.
    Args:
        mean (int): mean value of latent (time-domain)
        J: number of frequencies in complex 'DFT' matrix
    """
    # Number of terms for real-valued 'DFT' matrix
    Jv = J * 2
    return mean * (Jv / (2 * np.pi)) / 2

def logistic(x):
    return 1 / (1 + np.exp(-x))


def pickle_open(file):
    with open(file, "rb") as handle:
        data = pickle.load(handle)
    return data


def pickle_save(data, save_name):
    with open(save_name, "wb") as handle:
        pickle.dump(data, handle)
