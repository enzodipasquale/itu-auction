





def get_U_ij(self, v_j, i_idx ,j_idx = None):
        if j_idx is None:
            return α_ij[i_idx] +   np.min(N_k[None,None,:] + (1- τ_k[None,None,:])  * (γ_ij[i_idx,:,None] - v_j[None, :, None] ), axis= 2)