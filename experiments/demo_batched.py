import torch
from batched_itu_auction import ITUauction
import time

num_i = 1000
num_j = 980
num_t = 100



torch.manual_seed(234524354)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Φ_t_i_j = torch.stack([(torch.randint(0, 3, (num_i, 1), device=device) - torch.randint(0, 3, (1, num_j), device=device)) ** 2
                        for _ in range(num_t)], dim=0)

def get_U_t_i_j(v_t_j, t_id, i_id, j_id=None):
    if j_id is None:
        return Φ_t_i_j[t_id, i_id] - v_t_j[t_id]
    else:
        return Φ_t_i_j[t_id, i_id, j_id] - v_t_j

def get_V_t_i_j(u_t_i, t_id, j_id, i_id=None):
        if i_id is None:
            return Φ_t_i_j[t_id, :, j_id] - u_t_i[t_id]
        else:
            return Φ_t_i_j[t_id, i_id, j_id] - u_t_i
                            
market = ITUauction(num_i, num_j, num_t, get_U_t_i_j = get_U_t_i_j, get_V_t_i_j = get_V_t_i_j)

eps_init = 10
eps = 1e-3
scaling_factor = .5
tic = time.time()
# u_t_i, v_t_j, mu_t_i_j = market.forward_auction(eps = eps, return_mu_t_i_j = True)
# u_t_i, v_t_j, mu_t_i_j = market.reverse_auction(eps = eps, return_mu_t_i_j = True)
u_t_i, v_t_j, mu_t_i_j = market.forward_reverse_scaling(eps_init = eps_init, eps_target = eps, scaling_factor = scaling_factor)

market.check_equilibrium(u_t_i, v_t_j, mu_t_i_j, eps = eps)

toc = time.time()
print('-'*50)
print(f"Time taken for auction: {toc - tic:.4f} seconds")

# Check duality
primal_value_t = (Φ_t_i_j * mu_t_i_j).sum((1,2))
dual_value_t = u_t_i.sum(1) + v_t_j.sum(1)
print("Duality Gap: ", (dual_value_t - primal_value_t).amax())
