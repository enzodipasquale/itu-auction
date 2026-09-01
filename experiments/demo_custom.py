from time import perf_counter

import torch

from itu_auction.core import ITUauction


def main():
    torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surplus = torch.rand(32, 28, device=device)

    def get_u(v, i, j=None):
        if j is None:
            return surplus[i] - v.unsqueeze(0)
        return surplus[i, j] - v

    def get_v(u, j, i=None):
        if i is None:
            return surplus[:, j] - u.unsqueeze(1)
        return surplus[i, j] - u

    auction = ITUauction(32, 28, get_u, get_v, device=device)
    eps = 1e-3

    started = perf_counter()
    u, v, matching = auction.forward_reverse_scaling(
        1.0, eps, 0.5, certify_terminal=True
    )
    elapsed = perf_counter() - started

    cs, feasible, ir_i, ir_j = auction.check_equilibrium(u, v, matching, eps)
    print(f"device={device} elapsed={elapsed:.3f}s matched={matching.sum().item():.0f}")
    print(
        f"CS={cs.item():.3g} feasible={bool(feasible)} IR=({bool(ir_i)}, {bool(ir_j)})"
    )


if __name__ == "__main__":
    main()
