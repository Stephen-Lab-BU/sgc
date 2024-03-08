import math
import numpy as np

def sample_complex_normal(cov, n):
    m = cov.shape[0]
    L = np.linalg.cholesky(cov)
    temp = np.random.randn(m,n) + 1j*np.random.randn(m,n)
    
    sample = L @ temp
    
    return sample

def gen_complex_cov(K):
    A = np.random.randn(K,K) + 1j*np.random.randn(K,K)
    R = np.conj(A).T @ A
    
    return R

def sig_from_complex(c, time, freq):
    r = np.abs(c)
    theta = np.angle(c)
    # sig = r*np.exp(1j*2*np.pi*freq*theta*time)
    
    cos = np.array([math.cos(2*np.pi*freq*t + theta) for t in time])
    sin = np.array([math.sin(2*np.pi*freq*t + theta) for t in time])
    
    return r*(cos + 1j*sin)
    
def sig_from_real(a, b, time, freq):
    cos = np.array([a*math.cos(2*np.pi*freq*t) for t in time])
    sin = np.array([b*math.sin(2*np.pi*freq*t) for t in time])
    sig = cos + sin
    return 1/2*sig