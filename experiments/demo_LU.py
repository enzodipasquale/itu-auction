from time import perf_counter

import torch

from itu_auction import get_template


def main():
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surplus = 8 * torch.rand(40, 36, device=device)
    transfer_rate = 0.5 + torch.rand(40, 36, device=device)
    auction = get_template("LU")(surplus, transfer_rate)

    eps = 1e-3
    started = perf_counter()
    u, v, matching = auction.forward_reverse_scaling(
        8.0, eps, 0.5, certify_terminal=True
    )
    elapsed = perf_counter() - started

    cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps)
    print(f"device={device} elapsed={elapsed:.3f}s matched={matching.sum().item():.0f}")
    print(
        f"CS={cs.item():.3g} feasible={bool(feasible)} IR=({bool(ir_i)}, {bool(ir_j)})"
    )


if __name__ == "__main__":
    main()
