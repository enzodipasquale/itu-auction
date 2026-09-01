from time import perf_counter

import torch

from itu_auction import get_template


def main():
    torch.manual_seed(19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = 2 * torch.rand(36, 32, device=device)
    gamma = 2 * torch.rand(36, 32, device=device)
    thresholds = torch.tensor([0.0, 0.25, 0.75, 1.5], device=device)
    rates = torch.tensor([0.0, 0.1, 0.2, 0.3], device=device)
    auction = get_template("convex_tax")(alpha, gamma, thresholds, rates)

    eps = 1e-3
    started = perf_counter()
    u, v, matching = auction.forward_reverse_scaling(
        4.0, eps, 0.5, certify_terminal=True
    )
    elapsed = perf_counter() - started

    cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps)
    print(f"device={device} elapsed={elapsed:.3f}s matched={matching.sum().item():.0f}")
    print(
        f"CS={cs.item():.3g} feasible={bool(feasible)} IR=({bool(ir_i)}, {bool(ir_j)})"
    )


if __name__ == "__main__":
    main()
