import torch
from itu_auction.core import ITUauction

def convex_tax_template(α_i_j, γ_i_j, t_k, τ_k):

    # n_k = torch.zeros_like(t_k)
    # for m in range(1,len(t_k)):
    #     n_k[m] = n_k[m-1] + (1-τ_k[m-1]) * (t_k[m] - t_k[m-1])
    # N_k = n_k - (1-τ_k) * t_k  
    Δt = t_k[1:] - t_k[:-1]
    n_k = torch.cat([torch.zeros(1), torch.cumsum((1 - τ_k[:-1]) * Δt, dim=0)])
    N_k = n_k - (1 - τ_k) * t_k

    τ_k = τ_k.unsqueeze(0).clone()
    N_k = N_k.unsqueeze(0).clone()

    def get_U_i_j(v_j, i_idx, j_idx = None):
        v_j = v_j.unsqueeze(0)
        if j_idx is None:
            return α_i_j[i_idx] + (N_k.unsqueeze(0) + (1- τ_k.unsqueeze(0)) * (γ_i_j[i_idx].unsqueeze(2) - v_j.unsqueeze(2))).min(-1).values
        else:
            return α_i_j[i_idx, j_idx] + (N_k + (1- τ_k) * (γ_i_j[i_idx, j_idx].unsqueeze(1) - v_j)).min(-1).values

    def get_V_i_j(u_i, j_idx, i_idx = None):
        u_i = u_i.unsqueeze(1)
        if i_idx is None:
            return γ_i_j[:, j_idx] +(N_k.unsqueeze(0) + α_i_j[:,j_idx].unsqueeze(2) - u_i.unsqueeze(2))/(1- τ_k.unsqueeze(0)).min(-1).values
        else:
            return γ_i_j[i_idx,j_idx] + ((N_k + α_i_j[i_idx,j_idx].unsqueeze(1) - u_i)/(1- τ_k)).min(-1).values


    num_i, num_j = α_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)


