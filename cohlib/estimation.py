import numpy as np


def estimate_coherence(xf,yf, mag_sq=True):
    """
    Estimate coherence for a single frequency range from observed complex coefs. 
    Args:
        xf: (n_trials,) array of complex coefficients signal 1
        yf: (n_trials,) array of complex coefficients signal 2
        mag_sq: (bool) optional - return mean-squared coherence 
    Returns:
        coh: coherence estimate
    """

    Sxy = xf * yf.conj()
    Sxx = xf * xf.conj()
    Syy = yf * yf.conj()

    if mag_sq:
        num = np.abs(Sxy.mean(0))**2
        denom = Sxx.mean(0).real * Syy.mean(0).real

    else:
        num = np.abs(Sxy.mean(0))
        a = np.sqrt(Sxx.mean(0).real)
        b = np.sqrt(Syy.mean(0).real)
        denom = a*b

    coh  = num/denom

    return coh

def thr_coherence(Gamma, mag_sq=True):
    """
    Calculate theoretical coherence from covariance matrices. 
    Args:
        Gamma: (n_freqs, 2, 2) array of complex bcn covariance matrices
    Returns:
        t_coh: (n_freqs,) array of coherence values
    """

    if mag_sq:
        num = np.abs(Gamma[:,0,1])**2
        a = np.abs(Gamma[:,0,0])
        b = np.abs(Gamma[:,1,1])

    else:
        num = np.abs(Gamma[:,0,1])
        a = np.abs(Gamma[:,0,0])
        b = np.abs(Gamma[:,1,1])
        a, b = np.sqrt(a), np.sqrt(b)

    denom = a*b
    t_coh = num/denom

    return t_coh


# DEP
def estimate_coherence1f(xf,yf, ms=True):
    """
    Estimate coherence for a single frequency range from observed complex coefs. 
    Args:
        x: (n_trials,) array of complex coefficients signal 1
        y: (n_trials,) array of complex coefficients signal 2
        ms: (bool) optional - return mean-squared coherence 
    Returns:
        coh: coherence estimate
    """

    Sxy = xf * yf.conj()
    Sxx = xf * xf.conj()
    Syy = yf * yf.conj()

    if ms:
        num = np.abs(Sxy.mean())**2
        denom = Sxx.mean().real * Syy.mean().real

    else:
        num = np.abs(Sxy.mean())
        a = np.sqrt(Sxx.mean().real)
        b = np.sqrt(Syy.mean().real)
        denom = a*b

    coh  = num/denom

    return coh

def thr_coherence1f(cov, ms=True):

    if ms:
        num = np.abs(cov[0,1])**2
        a = np.abs(cov[0,0])
        b = np.abs(cov[1,1])

    else:
        num = np.abs(cov[0,1])
        a = np.abs(cov[0,0])
        b = np.abs(cov[1,1])
        a, b = np.sqrt(a), np.sqrt(b)

    denom = a*b
    t_coh = num/denom

    return t_coh