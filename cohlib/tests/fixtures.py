import pytest
import numpy as np
import random

from numpy.random import multivariate_normal as mvn

# DC

# simulated data
@pytest.fixture(scope='session')
def real_normal_sample(mean=0, var=1, n=100, seed=None):
    # if seed is not None:
    #     set_seed(seed)

    return np.random.randn(n)*np.sqrt(var) + mean







# no DC 