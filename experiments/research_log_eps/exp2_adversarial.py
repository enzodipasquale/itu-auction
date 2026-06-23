"""
Experiment 2: Try adversarial / hard cases.
- LU(kappa) for large kappa (1000, 10000).
- Larger N.
- Compare eps-scaling vs fixed-eps to make sure scaling matters.
- Price-war structure: nearly-tied utilities.
"""
import sys, os, math, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrumented import make_TU, make_LU, CountingAuction
import torch


def run_scaling(market, eps_target, eps_init=50.0, theta=0.5):
    market._reset_counters()
    t0 = time.time()
    market.forward_reverse_scaling(eps_init, eps_target, theta)
    return market.total_iters, time.time() - t0


def run_fixed(market, eps):
    """Fixed-eps forward-only (cold start)."""
    market._reset_counters()
    t0 = time.time()
    market.forward_auction(eps=eps)
    return market.total_iters, time.time() - t0


def main():
    print("=== Experiment 2: adversarial / hard cases ===\n")
    eps_targets = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    print("## Larger kappa (eps-scaling)")
    print(f"{'case':22s} {'N':>4s} " + " ".join(f"{e:>10.0e}" for e in eps_targets))
    for N in [50, 100, 200]:
        for kappa in [1000, 10000]:
            iters_row = []
            for eps in eps_targets:
                its = []
                for s in [1, 2, 3]:
                    m, _, _, _ = make_LU(N, N, seed=s, beta_min=1.0, beta_max=kappa)
                    it, _ = run_scaling(m, eps)
                    its.append(it)
                iters_row.append(statistics.mean(its))
            print(f"LU(kappa={kappa:>6d})         {N:>4d} " + " ".join(f"{x:>10.1f}" for x in iters_row))
    print()

    print("## Fixed-eps (no scaling) on TU and LU(kappa=10)")
    print(f"{'case':16s} " + " ".join(f"{e:>10.0e}" for e in eps_targets))
    for name, mk in [("TU", lambda s: make_TU(50, 50, seed=s)[0]),
                     ("LU(kappa=10)", lambda s: make_LU(50, 50, seed=s, beta_min=1.0, beta_max=10)[0])]:
        iters_row = []
        for eps in eps_targets:
            its = []
            for s in [1, 2, 3]:
                m = mk(s)
                it, _ = run_fixed(m, eps)
                its.append(it)
            iters_row.append(statistics.mean(its))
        print(f"{name:16s} " + " ".join(f"{x:>10.1f}" for x in iters_row))
    print()

    print("## Price-war: all consumers tied on a few yogurts")
    # Construct: U_ij depends only on j for most pairs, with tiny i-specific perturbation.
    # This induces price wars on every yogurt.
    torch.manual_seed(42)
    N = 50
    base = torch.linspace(1.0, 10.0, N)  # yogurt-specific value
    perturb_scale = 0.01
    for kappa in [1, 10, 100]:
        beta = torch.rand(N, N) * (kappa - 1.0) + 1.0  # in [1, kappa]
        # Phi_{ij} ≈ base_j + small noise → many near-ties
        Phi = base.unsqueeze(0).expand(N, N) + torch.randn(N, N) * perturb_scale

        def make_market():
            def get_U(v_j, i_id, j_id=None):
                if j_id is None:
                    return Phi[i_id] - beta[i_id] * v_j.unsqueeze(0)
                return Phi[i_id, j_id] - beta[i_id, j_id] * v_j
            def get_V(u_i, j_id, i_id=None):
                if i_id is None:
                    return (Phi[:, j_id] - u_i.unsqueeze(1)) / beta[:, j_id]
                return (Phi[i_id, j_id] - u_i) / beta[i_id, j_id]
            return CountingAuction(N, N, get_U, get_V)

        iters_row = []
        for eps in eps_targets:
            m = make_market()
            it, _ = run_scaling(m, eps)
            iters_row.append(it)
        print(f"price-war kappa={kappa:>4d}   " + " ".join(f"{x:>10d}" for x in iters_row))
    print()


if __name__ == "__main__":
    main()
