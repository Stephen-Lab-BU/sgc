import numpy as np

# from cohlib.tests.fixtures import real_normal_sample

from cohlib.mvcn import sample_zs_from_Gamma
# from cohlib.sample import sample_complex_normal

def test_sampling():
    """
    Testing for generating complex multivariate Gaussians
    """
    np.random.seed(1)

    L = 5000
    J = 10
    D = 2

    Gamma = np.stack([np.eye(D) + np.zeros((D,D))*1j for k in range(J)])


    # rng1 = np.random.default_rng(seed)

    # rng2 = np.random.default_rng(seed)
    num_samples = int(L*J*D)
    real_var = 0.5*np.eye(D*2)
    real_normal_sample = np.random.multivariate_normal(np.zeros(D*2), real_var, int(num_samples/2)).flatten()

    z_samples = sample_zs_from_Gamma(Gamma, L)
    test_samples = np.stack([z_samples.real, z_samples.imag]).flatten()

    assert z_samples.shape[0] == L
    assert z_samples.shape[1] == D
    assert z_samples.shape[2] == J 

    assert(real_normal_sample.size == test_samples.size)
    assert np.isclose(test_samples.mean(), real_normal_sample.mean(), atol=1e-2)
    assert np.isclose(test_samples.var(), real_normal_sample.var(), atol=1e-2)




