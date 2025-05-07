import torch
from itu_auction import get_template
import time

# Example of using the TU template
num_i = 600
num_j = 700

for _ in range(200):
    torch.manual_seed(_)
    Φ_i_j = (torch.randint(0, 10, (num_i, 1)) - torch.randint(0, 10, (1,num_j ))) **2
    α_i_j = torch.randint(1, 10, (num_i, num_j))

    market = get_template("LU")(Φ_i_j, α_i_j)

    # Solve equilibrium
    eps = 1e-3
    scaling_factor = 0.5
    eps_init = 10
    tic = time.time()
    # u_i, v_j, mu_i_j = market.forward_auction(eps= eps, return_mu_i_j=True)
    # u_i, v_j, mu_i_j = market.reverse_auction(eps= eps, return_mu_i_j=True)
    u_i, v_j, mu_i_j = market.forward_reverse_scaling(eps_init, eps, scaling_factor)
    toc = time.time()
    print(f"Time taken for auction: {toc - tic:.4f} seconds")

    # Check equilibrium
    market.check_equilibrium(u_i, v_j, mu_i_j, eps)

    # # Check duality
    # dual = u_i.sum() + v_j.sum()
    # primal = (mu_i_j * Φ_i_j).sum()
    # print("\n=== Duality Check ===")
    # print(f"Dual value                : {dual.item():.4f}")
    # print(f"Primal value (total cost) : {primal.item():.4f}")


