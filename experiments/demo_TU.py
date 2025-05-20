import torch
from itu_auction import get_template
import time
# Example of using the TU template
num_i = 1000
num_j = 900

for iter in range(100):
    print(iter)
    torch.manual_seed(iter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Φ_i_j = (
        (torch.randint(0, 3, (num_i, 1), device=device) - torch.randint(0, 3, (1, num_j), device=device)) ** 2
        )
    market = get_template("TU")(Φ_i_j)
    # market.method = "GS"
    # market.sampling_rate = 0.1

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
    # print(f"Dual value    : {dual.item():.4f}")
    # print(f"Primal value  : {primal.item():.4f}")

