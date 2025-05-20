import torch
from itu_auction.core import ITUauction

def TU_template(Φ_i_j):
    
    def get_U_i_j(v_j, i_id, j_id=None):
        if j_id is None:
            return Φ_i_j[i_id] - v_j.unsqueeze(0)
        else:
            return Φ_i_j[i_id, j_id] - v_j

    def get_V_i_j(u_i, j_id, i_id=None):
        if i_id is None:
            return Φ_i_j[:, j_id] - u_i.unsqueeze(1)
        else:
            return Φ_i_j[i_id, j_id] - u_i

    num_i, num_j = Φ_i_j.shape
    return ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)
