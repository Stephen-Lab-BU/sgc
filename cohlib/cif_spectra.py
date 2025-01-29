import numpy as np
def lognormal_cov(mu, Sigma):
    D = Sigma.shape[0]
    Sigma_ln = np.zeros_like(Sigma)
    for i in range(D):
        for j in range(D):
            sig_ij = np.exp(mu[i] + mu[j] + 0.5*(Sigma[i,i] + Sigma[j,j])) * (np.exp(Sigma[i,j]) - 1)
            Sigma_ln[i,j] = sig_ij

    return Sigma_ln


def log_link_spectrum(mu, Gamma):
    # convert Gamma to full real
    # construct W mat
    # calculate Sigma_x
    # calculate Sigma_lambda
    # calculate Gamma_lambda
    pass