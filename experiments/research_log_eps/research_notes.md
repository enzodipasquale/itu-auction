# Research notes: does ITU ε-scaling have O(log(1/ε)) complexity?

## Iteration 1 (initial)

### Empirical findings (already done, summarized)

Experiments in `exp1_scaling_baseline.py` and `exp2_adversarial.py` measured iteration counts for the combined forward–reverse ε-scaling auction at varying ε_target on:

- TU (κ = 1).
- LU (linearly transferable utility), `U_ij(v) = α_ij − β_ij v` with β heterogeneous, κ ∈ {2, 5, 20, 100, 1000, 10000}.
- N ∈ {50, 100, 200}.
- "Price-war" structures with near-tied utilities.

**Strong empirical conclusion**: iteration count fits `a + b · log(1/ε)` with R² > 0.95 in every case tested, including κ = 10000 and price-war structures. Slope on log-log axes is near 0 (between −0.05 and −0.08). The `1/ε` linear hypothesis fits much worse (R² ≈ 0.4).

Concretely: from ε_target = 1e-2 to 1e-6 (4 decades), TU iterations grow ~2× and LU(κ=10000, N=50) grows ~1.4×. If the bound were `O(1/ε)` we'd expect ~10000× growth.

For LU(κ=10000, N=200): iterations grow from 507 (1e-3) to 615 (1e-6) — ~80 per natural log of 1/ε, about `0.4·N` per natural log. Total iterations ≈ `0.4·N · log(1/ε_target)`. With per-iteration cost O(N), total work ≈ `0.4 N² log(1/ε)`.

**This matches Bertsekas's `O(N² log)` rate for TU.** The empirical evidence strongly suggests log(1/ε) is the right rate for ITU as well, possibly under the same set of assumptions.

### Theoretical progress

**Result (provable upgrade of Conjecture 26).** Suppose `U_ij(v) = F_i(α_ij − v)` with `F_i` strictly increasing and bi-Lipschitz with constants `c, C` *uniformly across i* — i.e., the Pareto-frontier shape is **consumer-specific** (not single-F as in the current Conjecture 26, but more general than identical-F across all (i,j)). Then there is a change of variables that reduces ITU ε-CS to TU ε/c-CS:

  - Set `ũ_i := F_i^{-1}(u_i)`. The function `F_i^{-1}` is bi-Lipschitz with `1/C ≤ (F_i^{-1})' ≤ 1/c`.
  - ITU ε-CS: `u_i ≥ F_i(α_ij − v_j) − ε` ⇔ `F_i^{-1}(u_i + ε) ≥ α_ij − v_j`.
  - Using `F_i^{-1}(u_i + ε) ≤ ũ_i + ε/c`: gives `ũ_i + v_j ≥ α_ij − ε/c`, which is TU ε/c-CS in `(ũ, v)` coordinates with surplus `α_ij`.
  - Conversely TU ε'-CS ⇒ ITU `Cε'`-CS via `u_i = F_i(ũ_i) ≥ F_i(α_ij − v_j − ε') ≥ U_ij(v_j) − Cε'`.

So ITU ε-CS in `(u, v)` ⇔ TU ε'-CS in `(ũ, v)` with `ε' ∈ [ε/C, ε/c]`. The change of variables is well-defined (each F_i is invertible) and is bi-Lipschitz on the relevant domain.

The auction's bid update `v_{j_i} ← V_{ij_i}(w_i − ε) = α_{ij_i} − F_i^{-1}(w_i − ε)`, in TU coordinates, becomes `v_{j_i} ← α_{ij_i} − w̃_i + ε̃` where `w̃_i = F_i^{-1}(w_i)` and `ε̃ = F_i^{-1}(w_i) − F_i^{-1}(w_i − ε) ∈ [ε/C, ε/c]`. So the auction in `(ũ, v)` coordinates is a TU forward auction with **variable** bid increment `ε̃` lower-bounded by `ε/C`.

By the standard Bertsekas analysis (TU price-stability lemma, per-phase O(N²) bound), variable `ε̃ ≥ ε/C` is enough: each bid raises some `v_j` by at least `ε/C`, the price profile during a phase stays within `O(N · ε_phase / c)` of any other ε_phase-CS profile (TU lemma applied with the ε̃ that's active at the phase), and the per-phase iteration count is `O(N² · κ)`. Summing over phases: `O(N² · κ · log(1/ε_target))` iterations, `O(N³ · κ · log(1/ε_target))` work.

**This proves O(log(1/ε)) for the consumer-specific-F class.** Strictly more general than what's in the paper (Conjecture 26 with single F).

The class includes:
- TU: `F_i(z) = z` for all i.
- Single-F additive: `F_i = F` common.
- Heterogeneous CRRA risk aversion: `F_i(z) = (z + c)^{1−ρ_i}/(1−ρ_i)` with consumer-specific `ρ_i`. (Important application: BGHOS Example 2.2.)
- Heterogeneous logit-like smoothing.

The class does NOT include:
- LU with `β_ij` heterogeneous in *both* indices (because the slope of the frontier on `(i,j)` depends on `j` not just `i`).
- Convex-tax with kink locations that depend on `j`.

For these excluded cases, the empirical evidence still shows log scaling, so the result is conjecturally tighter.

### Why the path-amplification proposition (Prop 25 in paper) is loose

Prop 25 chains `δ_l ≤ κ · δ_{l−1} + 2ε/c` along an alternating path, giving `κ^N · ε`. Empirically the bound is `O(Nε)`. The proposition is loose because:

1. It uses **worst-case κ** at each step, not the actual `β` realized at the step.
2. It loses information about closed cycles in `M Δ M*`. On a closed cycle, the product of amplification factors equals 1 (since j's repeat), so the local amplifications "cancel" in some weighted sense. Prop 25 doesn't exploit this.
3. The auction's algorithmic trajectory does not in fact follow worst-case paths — bid increments are always relative to the actual U' at the bid point, not the worst-case κ.

These suggest the path argument is the wrong proof technique. A successful proof would likely use either (a) the convex-duality reformulation (Galichon) which gives an LP-like structure, or (b) a direct potential function argument on the algorithm's state.

### Action items for next iteration

- [x] Write up findings in this file.
- [ ] Update paper: convert Conjecture 26 (single-F additive) into a Theorem (consumer-specific F_i additive). Restate Conjecture 26 to address the harder case (general bi-Lipschitz, e.g., LU with `β_ij` heterogeneous).
- [ ] Commit/push.
- [ ] (Future iterations) Theoretical attack on the harder case via convex duality. Try the GKW (2019) sup-convolution structure.
- [ ] (Future iterations) Construct a *provable* counterexample for ω(log) scaling, or convince ourselves this is impossible.


---

## Iteration 2

### Bug found: Lemma 24 (TU price stability) was empirically false as stated

The lemma I had claimed `||v - v*||_∞ ≤ 2Nε` for any two ε-CS profiles. Direct measurement (`exp3_price_stability.py`):

- TU N=100, ε ∈ [1e-5, 1e-1]: `||v_fwd - v_rev||_∞ ≈ 9.3` for ALL ε. Constant, not O(Nε).
- LU κ=1000 N=100: ≈ 1.6, also constant in ε.

Two arbitrary ε-CS profiles can be at the equilibrium-set diameter apart, which is independent of ε. Bertsekas's actual lemma is about the algorithm's per-phase price movement (between consecutive phases of the same scaling run), not arbitrary ε-CS comparisons. **The original Lemma 24 was a wrong recollection.**

The corrected statement (now in the paper) is:
> *Per-phase movement.* Starting a phase at parameter ε_new from prices that came from the prior phase at ε_old, the algorithm's prices move by at most O(N ε_old).

This is what powers the polytime bound. In TU it's proved via LP duality (the LP-dual of the assignment).

### Convex duality angle was wrong direction

User correctly pointed out: ITU has no LP / convex-duality reformulation in general. Galichon-Jacquet (2024, arXiv:2402.12200) make this explicit: "LTU matching problems can also be reframed as games, but **not** as linear programs." The TU-style proof technique cannot be ported to ITU.

The right angle is **monotonicity / Tarski / lattice**, since ITU equilibria are characterized as fixed points of an isotone operator (Crawford-Knoer 1981, Quinzii 1984, Demange-Gale 1985, Kaneko 1982, Alkan 1989/1990, Adachi 2000, Hatfield-Milgrom 2005). The full lattice structure is well known.

### Literature update

The paper's Relation-to-the-Literature section was wrong: it credited GKW (2019) with introducing ITU, when in fact ITU as a framework (with continuous decreasing transfer functions, reservation prices, lattice-structured equilibria, finite-step algorithms for piecewise-linear preferences) was developed in the 1980s — Crawford-Knoer 1981, Kaneko 1982, Quinzii 1984, Demange-Gale 1985, Alkan 1989, Alkan-Gale 1990. Now correctly cited.

### What I now believe the conjecture reduces to

The polytime bound for general bi-Lipschitz ITU reduces to:

**Conjecture (per-phase price movement).** Let `(u^in, v^in)` satisfy `(ε_old-CS)`. Run one forward phase at `ε_new ≤ ε_old` starting from this `(u^in, v^in)` (matching reset). Then `||v^out - v^in||_∞ ≤ O(N ε_old / c)`.

If this holds, combined with the bid-increment lemma, it gives `O(N² κ)` bids per phase, hence `O(N³ κ log(1/ε_target))` work — exactly the conjectured polytime bound.

Empirically the conjecture holds: per-phase movement scales as `Θ(ε_phase)` not `Θ(D)`.

The proof must go through the lattice / order structure, not LP duality. The diameter of the *full* `ε`-equilibrium lattice can be `Θ(D)` (independent of `ε`), per the path-amplification proposition. The conjecture is precisely that *the auction's actual trajectory stays in a much smaller order interval* than the full lattice diameter, where the smaller interval has width `O(Nε_old)`. This is plausible because the auction is constrained by isotonicity: forward bids only raise prices, reverse bids only raise utilities. But making it precise is the open question.

### Action items for iteration 3+

- [ ] Try to formalize the per-phase movement conjecture using the Demange-Gale lattice and isotone operator T from BGHOS Section 4.2.2 / Adachi 2000.
- [ ] Specifically: characterize the auction's trajectory as iteration of a contractive operator under a suitable order-aware metric.
- [ ] If successful: prove Conjecture 28 in full generality. If not, propose a STRONGER assumption (more general than consumer-specific additivity) under which the proof goes through.


---

## Iteration 4

### Distinguishing "ITU has no LP" carefully

User cited Galichon-Jacquet (2024) as evidence that ITU has no LP. Re-reading: they show the *assignment* π in LU/ITU cannot be characterized as the optimum of an LP (no totally-unimodular primal). But the *dual side* — equilibrium (u,v) for LU — is still defined by linear inequalities `u_i + β_ij v_j ≥ α_ij - ε`. So for **LU specifically**, there is a (u,v)-LP, even though there is no π-LP.

This raises the question: can the LP-on-(u,v) for LU give us Conjecture 30 (per-phase movement) for LU specifically, even though it doesn't for general bi-Lipschitz ITU?

### Attempted LP-sensitivity argument for LU

Setup: LU with `β_ij ∈ [β_min, β_max]`, ε-CS = `u_i + β_ij v_j ≥ α_ij - ε`.

Claim attempt: any ε-feasible (u,v) has a 0-feasible (u, v^0) within `O(Nε/β_min)` componentwise.

Construction: set `v^0_j := v_j + Nε/β_min`. Check feasibility:
```
u_i + β_ij v^0_j = u_i + β_ij v_j + N β_ij ε / β_min
                 ≥ (α_ij - ε) + N β_ij ε / β_min
                 ≥ α_ij - ε + N ε        (using β_ij/β_min ≥ 1)
                 ≥ α_ij                  (for N ≥ 1)
```
✓. So `||v^0 - v||_∞ ≤ Nε/β_min`.

Implication: ε-feasible set ⊂ (0-feasible set) + `Nε/β_min`-box.

### But this does NOT give Conjecture 30

The argument shows ε-feasible profiles are *within* `Nε/β_min` of 0-feasible profiles. But the *0-feasible set itself has diameter Θ(D) independent of ε* (lattice of equilibria; experimentally verified ~9.3 for TU N=100). So two arbitrary ε-feasible profiles can differ by `D + 2Nε/β_min`, dominated by `D`.

Per-phase movement (Conjecture 30) is *not* about arbitrary ε-feasible profiles. It's about the algorithm's specific trajectory. The argument above is too coarse — it only bounds distance to the 0-feasible set, not distance between two specific algorithmic profiles.

### What's actually needed

For per-phase movement, we need to show the auction's trajectory does **not** traverse the full 0-feasible polytope diameter in any single phase. The auction starts at `(u^in, v^in)` ε_old-feasible (close to 0-feasible profile P^in) and ends at `(u^out, v^out)` ε_new-feasible (close to 0-feasible profile P^out). We need P^in ≈ P^out, or at least `||P^in - P^out||_∞ = O(Nε_old)`.

Why might this be true? **Monotonicity of the algorithm**: forward iterations only raise v, so the auction's trajectory tracks a *single isotone path* through the equilibrium lattice. Different phases trace different points along this path, but consecutive phases produce points that are close because (a) ε only decreased a little and (b) the algorithm's bidding rule constrains how far v can rise before terminating.

This is the right intuition, but I don't have a rigorous bound. Specifically, claim (b) — "the auction's bidding rule constrains v rise" — would need to be derived from the algorithm's specific structure. Bertsekas's TU argument does this via the LP-dual structure of the assignment problem; for LU/ITU we need an analogue without unimodularity.

### Status of Conjecture 30

- Holds for TU (Bertsekas).
- Holds for consumer-specific additivity (TU reduction; Theorem 26).
- Open for LU with `β_ij` heterogeneous in both indices.
- Open for general bi-Lipschitz ITU.

The LP structure on (u,v) for LU is necessary but not sufficient to lift Bertsekas's argument; we'd also need to handle the absence of integer/unimodular vertex structure on the matching side.

### Action for next iteration

- [ ] Look for explicit Bertsekas (1991, Sec 4.5) statement of the per-phase price-movement lemma; check whether the proof uses unimodularity essentially or just LP sensitivity.
- [ ] If only LP sensitivity is needed, the argument should extend to LU (where (u,v)-LP exists).
- [ ] Alternatively: try to construct a candidate adversarial LU instance where per-phase movement is `Θ(D)`, not `O(Nε_old)`. This would refute Conjecture 30 for LU. (Note: empirics in Iteration 1 suggest such instances may not exist.)


---

## Iteration 5

### Counterexample to Conjecture 30 also applies in TU

The 2x2 LU counterexample I found in iteration 4 also works in TU (set β = 1). So **Bertsekas's TU per-phase price-movement lemma (Lemma 24 in my paper) is for the matching-PRESERVE variant**, not matching-reset. With matching-reset, even TU can show overshoot from a 0-CS state.

This means Theorem 26's proof — which cites Bertsekas's TU bound — is implicitly for the matching-preserve variant, even though my Section 3.3 algorithm is matching-reset.

### Hedge added to Theorem 26

In iteration 5 I added a hedge to Theorem 26's proof: "Bertsekas's $O(N^2)$ per-phase bound applies cleanly to the matching-preserve variant of ε-scaling … Theorem 26 as stated is therefore for the matching-preserve variant; the matching-reset algorithm of Section 3.3 matches the asymptotic empirically but its formal analysis under matching-reset reduces to Conjecture (per-phase bid count)."

### Attempted argument for "# bids per consumer per phase ≤ N"

For a forward phase: v is monotone non-decreasing. Each bid by consumer i raises v_{j_i}, lowering U_ij_i(v) by some amount. After the bid, j_i is no longer first-best for i.

**Naive argument**: i never bids on the same j twice in a phase (because v_j only goes up, j becomes less attractive). Hence # bids per consumer ≤ N.

**Why the naive argument fails**: between i's bids on different yogurts, OTHER yogurts' prices can also rise (other consumers' bids). So when i is displaced and re-bids, the relative ranking of yogurts can have changed, and j_i might be first-best again.

In TU, Bertsekas's actual argument uses the LP-dual: total profit Σπ_i has bounded movement, each forward bid by i decreases π_i by ≥ ε, hence # bids per consumer is bounded by (LP-gap)/ε = O(N).

In ITU (general bi-Lipschitz), no LP-dual. The naive argument doesn't extend, and a Bertsekas-style proof would need a substitute potential function. This is the open problem.

### Status going forward

The clearest open question is now: prove that **in matching-reset ε-scaling for ITU, # bids per consumer per phase is O(N · poly(κ))** — the ITU analog of Bertsekas's TU per-consumer bound. Empirically holds; theoretical proof open.

The matching-PRESERVE variant might be easier to analyze, as the price-movement lemma may transfer cleanly via the change of variables in Theorem 26's proof.

### Next iteration

- Consider rewriting Section 3.3 to describe matching-PRESERVE as the primary algorithm (with a remark that matching-reset is the practical variant).
- Or: specifically attempt the per-consumer-bid bound using a monotonicity-based potential function.


---

## Iteration 6

### I had a wrong mental model of the per-bid increment

In iterations 4-5 I claimed Bertsekas's TU bound is "for matching-preserve only" because in matching-reset, the auction can overshoot. Re-examining: the overshoot is in **price movement**, not in **bid count**.

In TU forward iteration: consumer `i` finds first-best `j_i` with utility `ψ_i(j_i)` and second-best `w_i`. Bid is `b = α_{i,j_i} - (w_i - ε) = α_{i,j_i} - w_i + ε`. Per-bid increment in `v_{j_i}` is `b - v_{j_i}^old = (α_{i,j_i} - w_i + ε) - v_{j_i}^old = ψ_i(j_i) - w_i + ε`.

So per-bid increment is `(first-best − second-best) + ε`. This can be **much larger than ε** when first-best and second-best are far apart.

**Implication**: even in TU, the per-bid increment is variable. Bertsekas's lower bound `≥ ε` is not tight upper-bound; bid count is **not** simply `movement / ε`.

This invalidates my earlier reasoning that "Bertsekas's bound is matching-preserve only." Bertsekas's per-phase bid-count bound `O(N²)` likely holds in both variants — it's about how many *bids* happen, not how much *movement* happens. The price-stability lemma I cited (Lemma 24) was about price movement, but Bertsekas's actual polynomial-time argument bounds bid count via a different potential (decreasing profit `π`).

### So Theorem 26 might be OK after all

If Bertsekas's bound is `O(N²)` bids per phase regardless of variant, then Theorem 26 applies to matching-reset too. The hedge I added in iteration 5 was overly cautious.

But: I haven't actually verified Bertsekas's exact argument for matching-reset. The decreasing-`π` potential argument requires `π` to be tracked across phases, which the user's code doesn't do (it recomputes `u` at end of each forward call). For matching-PRESERVE, `π` is preserved naturally.

For matching-RESET, the algorithm doesn't track `π` between phases — but it does track `v`. If `v` from phase k is at the ε_k-equilibrium and we run phase k+1 with smaller ε, the new u's are computed from v during the iterations. Whether the bid-count bound holds without explicit π-tracking is unclear.

**Net**: I'm not confident either way. Theorem 26 as stated is plausible but the proof has a gap I can't close definitively without re-deriving Bertsekas's argument carefully.

### What I'll do for the paper

Leave the iteration-5 hedge in place. It's defensive but honest. The substantive open question is still Conjecture 28 (general log scaling for ITU), which is independent of this matching-reset/preserve issue.

### Open questions list (status)

- **Conjecture 28** (general log scaling for bi-Lipschitz ITU): open. Empirically holds.
- **Conjecture 30** (per-phase bid count `O(N² poly(κ))`): open. Stated as the key sub-conjecture.
- **Theorem 26** (log scaling under consumer-specific additivity): proved modulo Bertsekas's TU bound. With hedge, applies to matching-preserve variant.
- **Theorem 18** (polytime `O(N²/ε)` for bi-Lipschitz): proven directly for matching-reset. ✓

### Action for future iterations

I'm at a research dead end without external reference material (specifically, Bertsekas 1991 Section 4.5 to see what the per-phase bid-count argument actually is). Without it I can't:
1. Verify Theorem 26's proof in detail.
2. Make progress on the per-phase bid-count conjecture for ITU.

A productive direction would be reading Bertsekas (1991) directly. My local copy is `resources/LNets_Full_Book.pdf`. Iteration 7 should look at that.


---

## Iteration 6 (continued): reading Bertsekas (1991) directly

Found the relevant section: **Section 4.1.3, p.174, "Computational Aspects – ε-Scaling"** in Bertsekas's *Linear Network Optimization* (1991). The actual claim:

> "For **integer data**, it can be shown that the worst-case running time of the auction algorithm using scaling and appropriate data structures is `O(nA log(nC))`; see [BeE88], [BeT89]."

The qualification "**for integer data**" is critical. For continuous data, Bertsekas explicitly notes:

- **Exercise 1.4(a)**: per-object bid bound is `1 + C/ε` for dense problems starting from zero prices — i.e., `C/ε`, not log.
- **Exercise 1.4(b)** [Castañón 1991]: a worst-case construction (Figure 1.2) where some objects receive `nC/ε` bids — quadratically worse than the conjectured log scaling.

**Implication**: the log scaling I've been claiming for continuous TU (and using in Theorem 26's proof for the analogous continuous ITU case) is *not* a rigorous result in Bertsekas. It's a folk extension from integer data, supported by empirics. The integer log bound exists; continuous log bound is conjectural.

For my paper:
- The "polytime log scaling" Theorem 26 attributes to Bertsekas was ALREADY an extrapolation from integer to continuous data.
- The continuous case is empirically log but no rigorous proof in TU itself.
- So my matching-reset-vs-preserve hedge in iteration 5 is a *secondary* concern; the *primary* gap is integer-vs-continuous.

This actually strengthens the case for Conjecture 28 being open even at the TU level, restricted to continuous data.

### Revised understanding

The ε-scaling auction's complexity hierarchy is:

| Setting | Bound | Status |
|---|---|---|
| TU, integer data | `O(N³ log(NC))` | **Theorem (Bertsekas-Eckstein 1988)** |
| TU, continuous bounded data | `O(N³ log(D/ε_target))` | Folk extension; rigorously, `O(N² C/ε_target)` worst-case (Cas91) |
| ITU consumer-specific additive, continuous | same as TU continuous | Folk via change of variables |
| ITU general bi-Lipschitz, continuous | open | Conjecture 28 |

**Honest paper status**: even in TU continuous, the log-scaling bound is folk. My Theorem 26 inherits this status. Conjecture 28 (general bi-Lipschitz) is a *further* open question on top of an already-folk continuous-TU baseline.

I should update the paper to be more precise about this.
