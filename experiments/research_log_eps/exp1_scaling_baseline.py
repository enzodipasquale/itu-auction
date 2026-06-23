"""
Experiment 1: how does iteration count scale with 1/eps_target?
TU and LU(kappa) cases. Theory predicts:
  TU (Bertsekas): O(N^2 log(1/eps))
  ITU bound (our paper):  O(N^2 / eps)
  ITU truth (open):       ?
"""
import sys, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instrumented import make_TU, make_LU
import torch
import time


def run(market, eps_target, eps_init=50.0, theta=0.5):
    market._reset_counters()
    t0 = time.time()
    u, v, mu = market.forward_reverse_scaling(eps_init, eps_target, theta)
    dt = time.time() - t0
    return {
        "fwd_per_phase": list(market.fwd_iters_per_phase),
        "rev_per_phase": list(market.rev_iters_per_phase),
        "total_iters": market.total_iters,
        "num_phases": market.num_phases,
        "time": dt,
    }


def fit_loglog(xs, ys):
    """Return slope of log(y) vs log(x) — does y scale as x^slope?"""
    import numpy as np
    lx = np.log(np.array(xs))
    ly = np.log(np.array(ys))
    # least squares
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope


def fit_log_vs_lin(xs, ys):
    """Compare two hypotheses:
       H1: y = a + b * log(1/eps)     (log scaling)
       H2: y = a + b * (1/eps)        (linear scaling)
       Return R^2 for each.
    """
    import numpy as np
    inv = 1.0 / np.array(xs)
    log_inv = np.log(inv)
    y = np.array(ys)
    def r2(x):
        b = np.polyfit(x, y, 1)
        yhat = np.polyval(b, x)
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2(log_inv), r2(inv)


def main():
    eps_targets = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    N = 50
    seeds = [1, 2, 3]

    cases = []
    cases.append(("TU", lambda s: make_TU(N, N, seed=s)[0]))
    for kappa in [2, 5, 20, 100]:
        beta_max, beta_min = kappa, 1.0
        cases.append((f"LU(kappa={kappa})",
                      lambda s, kmax=beta_max, kmin=beta_min:
                      make_LU(N, N, seed=s, beta_min=kmin, beta_max=kmax)[0]))

    print(f"# Iteration count vs eps_target  (N={N})\n")
    print(f"{'case':16s} {'eps_target':>11s} {'total_iter':>11s} {'fwd_phases':>11s} {'rev_phases':>11s} {'time(s)':>9s}")
    rows = {}
    for name, mk in cases:
        rows[name] = []
        for eps in eps_targets:
            iters = []
            t = 0.0
            for s in seeds:
                m = mk(s)
                r = run(m, eps)
                iters.append(r["total_iters"])
                t += r["time"]
            avg_iter = sum(iters) / len(iters)
            rows[name].append((eps, avg_iter))
            print(f"{name:16s} {eps:11.1e} {avg_iter:11.1f} "
                  f"{len(r['fwd_per_phase']):11d} {len(r['rev_per_phase']):11d} {t/len(seeds):9.3f}")
        print()

    # Analyze scaling
    print("\n# Scaling analysis: y = total_iter, x = 1/eps_target")
    print(f"{'case':16s} {'slope(loglog)':>15s} {'R^2(log(1/eps))':>17s} {'R^2(1/eps)':>12s} {'verdict':>20s}")
    for name, data in rows.items():
        xs = [eps for eps, _ in data]
        ys = [y for _, y in data]
        slope = fit_loglog(xs, ys)
        r2_log, r2_lin = fit_log_vs_lin(xs, ys)
        if r2_log > r2_lin + 0.05:
            verdict = "log(1/eps) wins"
        elif r2_lin > r2_log + 0.05:
            verdict = "1/eps wins"
        else:
            verdict = "tied"
        # slope: with y = (1/eps)^s, slope of loglog is s. s near 0 = log; s near 1 = linear.
        print(f"{name:16s} {slope:15.3f} {r2_log:17.3f} {r2_lin:12.3f} {verdict:>20s}")

    return rows


if __name__ == "__main__":
    main()
