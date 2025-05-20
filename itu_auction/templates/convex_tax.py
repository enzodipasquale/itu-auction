import torch
from itu_auction.core import ITUauction

def convex_tax_template(α_i_j, γ_i_j, t_k, τ_k):

    Δt = t_k[1:] - t_k[:-1]
    n_k = torch.cat([torch.zeros(1), torch.cumsum((1 - τ_k[:-1]) * Δt, dim=0)])
    N_k = n_k - (1 - τ_k) * t_k

    τ_k = τ_k.unsqueeze(0).clone()
    N_k = N_k.unsqueeze(0).clone()

    def get_U_i_j(v_j, i_id, j_id = None):
        if j_id is None:
            v_j = v_j.unsqueeze(0)
            return α_i_j[i_id] + (N_k.unsqueeze(0) + (1- τ_k.unsqueeze(0)) * (γ_i_j[i_id] - v_j).unsqueeze(2)).amin(-1)
        else:
            return α_i_j[i_id, j_id] + (N_k + (1- τ_k) * (γ_i_j[i_id, j_id] - v_j).unsqueeze(1)).amin(-1)

    def get_V_i_j(u_i, j_id, i_id = None):
        if i_id is None:
            u_i = u_i.unsqueeze(1)
            return γ_i_j[:, j_id] + ((N_k.unsqueeze(0) + (α_i_j[:,j_id] - u_i).unsqueeze(2))/(1- τ_k.unsqueeze(0))).amin(-1)
        else:
            return γ_i_j[i_id,j_id] + ((N_k + (α_i_j[i_id,j_id] - u_i).unsqueeze(1))/(1- τ_k)).amin(-1)


    num_i, num_j = α_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)