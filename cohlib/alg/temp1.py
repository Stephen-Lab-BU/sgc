def construct_Gamma_full_real_dc(DC_update, Gamma_update_complex, K, num_J_vars, invert=False):
    J = int((num_J_vars-1)/2)


    Gamma_full = np.zeros((K*num_J_vars, K*num_J_vars))
    if invert is True:
        DC_update = np.linalg.inv(DC_update)
    base_filt = np.zeros(num_J_vars)
    base_filt[0] = 1
    j_filt = np.tile(base_filt.astype(bool), K)

    for k in range(K):
        Gamma_full[j_filt,k*num_J_vars] = DC_update[:,k]



    for j in range(J):
        Gamma_n = Gamma_update_complex[j,:,:]
        if invert is True:
            Gamma_n = np.linalg.inv(Gamma_n)
        Gamma_n_real = reverse_rearrange_mat(transform_cov_c2r(Gamma_n),K)
        base_filt = np.zeros(num_J_vars)
        j_var = int(j*2 + 1)
        base_filt[j_var:j_var+2] = 1
        j_filt = np.tile(base_filt.astype(bool), K)
        # print(j_filt)
        for k in range(K):
            kj = int(k*2)
            Gamma_full[j_filt,k*num_J_vars+j_var:k*num_J_vars+j_var+2] = Gamma_n_real[:,kj:kj+2]

    return Gamma_full