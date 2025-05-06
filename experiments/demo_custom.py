from itu_auction.core import ITUauction
import torch
import time

# Example of using the TU template
num_i = 15
num_j = 13

torch.manual_seed(849849)
Φ_i_j = (torch.randint(0, 9, (num_i, 1)) - torch.randint(0, 9, (1,num_j ))) **2


def get_U_i_j(v_j, i_id, j_id=None):
    if j_id is None:
        return Φ_i_j[i_id] - v_j.unsqueeze(0)
    else:
        return Φ_i_j[i_id, j_id] - v_j

def get_V_i_j(u_i, j_id, i_id=None):
    if i_id is None:
        return Φ_i_j[:, j_id] - u_i.unsqueeze(0)
    else:
        return Φ_i_j[i_id, j_id] - u_i

market = ITUauction(num_i, num_j, get_U_i_j, get_V_i_j)


# Solve 
eps = 1e-3
scaling_factor = 0.5
eps_init = 10
tic = time.time()
u_i, v_j, mu_i_j = market.forward_auction(eps= eps, return_mu_i_j=True)
# u_i, v_j, mu_i_j = market.reverse_auction(eps= eps)
# u_i, v_j, mu_i_j = market.forward_reverse_scaling(eps_init, eps, scaling_factor)
toc = time.time()
print(f"Time taken for auction: {toc - tic:.4f} seconds")

# Check equilibrium
market.check_equilibrium(u_i, v_j, mu_i_j, eps)

# Check duality
# dual = u_i.sum() + v_j.sum()
# primal = (mu_i_j * Φ_i_j).sum()
# print("\n=== Duality Check ===")
# print(f"Dual value                : {dual.item():.4f}")
# print(f"Primal value (total cost) : {primal.item():.4f}")
