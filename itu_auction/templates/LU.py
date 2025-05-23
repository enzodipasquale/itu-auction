import torch
from itu_auction.core import ITUauction

def LU_template(Φ_i_j, α_i_j):
    
    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return Φ_i_j[i_id] - α_i_j[i_id] * v_j.unsqueeze(0)
        else:
            return Φ_i_j[i_id, j_id] - α_i_j[i_id, j_id] * v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return (Φ_i_j[:, j_id] - u_i.unsqueeze(1)) / α_i_j[:, j_id]
        else:
            return (Φ_i_j[i_id, j_id] - u_i) / α_i_j[i_id, j_id]

    num_i, num_j = Φ_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)