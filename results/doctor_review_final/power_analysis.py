"""
Power Analysis for Doctor Review Experiment
============================================
Primary contrast: Pipeline (Qwen-DA) vs Qwen Baseline
Primary endpoint: Accuracy (single endpoint, no Holm correction)
Secondary: ChatGPT comparison is descriptive only (no formal test)

Monte Carlo simulation using Wilcoxon signed-rank test
on paired scenario-level rater-mean differences.

Usage:
    python3 power_analysis.py
"""
import numpy as np
from scipy import stats
from collections import Counter

np.random.seed(42)

# ── Empirical OR estimation from LLM judge data ──────────────────────

def estimate_or():
    """Estimate proportional-odds OR from LLM judge ACC scores (20 breast samples)"""
    pipeline_acc = [5,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
    baseline_acc = [5,5,5,4,3,4,5,4,4,4,5,4,4,4,4,4,4,4,5,3]

    p_dist = Counter(pipeline_acc)
    b_dist = Counter(baseline_acc)

    print("=== Empirical ACC Distribution (LLM Judge, 20 breast) ===")
    print(f"Score  Pipeline  Baseline")
    for s in [1,2,3,4,5]:
        print(f"  {s}      {p_dist.get(s,0):2d}        {b_dist.get(s,0):2d}")
    print(f"Mean:  {np.mean(pipeline_acc):.2f}      {np.mean(baseline_acc):.2f}")

    # Paired comparison
    wins = sum(1 for p, b in zip(pipeline_acc, baseline_acc) if p > b)
    losses = sum(1 for p, b in zip(pipeline_acc, baseline_acc) if p < b)
    ties = sum(1 for p, b in zip(pipeline_acc, baseline_acc) if p == b)
    print(f"\nPaired: Pipeline wins={wins}, Ties={ties}, Baseline wins={losses}")

    # Mann-Whitney OR
    n = len(pipeline_acc)
    superior = sum(1 for p in pipeline_acc for b in baseline_acc if p > b)
    inferior = sum(1 for p in pipeline_acc for b in baseline_acc if p < b)
    tied = n*n - superior - inferior
    prob_sup = (superior + 0.5*tied) / (n*n)
    prob_inf = (inferior + 0.5*tied) / (n*n)
    mw_or = prob_sup / prob_inf

    # Wilcoxon p-value
    diffs = [p-b for p,b in zip(pipeline_acc, baseline_acc) if p != b]
    _, p_val = stats.wilcoxon(diffs)

    print(f"Mann-Whitney OR = {mw_or:.2f}")
    print(f"Wilcoxon p = {p_val:.4f}")
    print(f"\nLLM judge OR ≈ {mw_or:.1f}. Human raters conservative estimate: OR = 2.0")
    return mw_or


# ── Power calculation ─────────────────────────────────────────────────

# Baseline ACC distribution assumption
# From LLM judge: ~10% score 3, ~60% score 4, ~30% score 5
BASELINE_PROBS = np.array([0.00, 0.00, 0.10, 0.60, 0.30])


def shift_probs(probs, log_or):
    """Shift ordinal distribution by proportional-odds OR."""
    cum = np.cumsum(probs)
    shifted_cum = []
    for c in cum[:-1]:
        if c <= 0 or c >= 1:
            shifted_cum.append(c)
        else:
            logit = np.log(c / (1 - c))
            new_logit = logit + log_or
            shifted_cum.append(1 / (1 + np.exp(-new_logit)))
    shifted_cum.append(1.0)
    shifted_probs = np.diff([0] + shifted_cum)
    shifted_probs = np.maximum(shifted_probs, 0)
    shifted_probs /= shifted_probs.sum()
    return shifted_probs


def simulate_power(n_scenarios, n_raters, OR, n_sims=5000, alpha=0.05):
    """Monte Carlo power for single endpoint, paired Wilcoxon test."""
    log_or = np.log(OR)
    pipeline_probs = shift_probs(BASELINE_PROBS, log_or)
    scores = [1, 2, 3, 4, 5]

    rejections = 0
    for _ in range(n_sims):
        scenario_diffs = []
        for _ in range(n_scenarios):
            base_ratings = np.random.choice(scores, size=n_raters, p=BASELINE_PROBS)
            pipe_ratings = np.random.choice(scores, size=n_raters, p=pipeline_probs)
            scenario_diffs.append(np.mean(pipe_ratings) - np.mean(base_ratings))

        diffs = np.array(scenario_diffs)
        nonzero = diffs[diffs != 0]
        if len(nonzero) < 2:
            continue
        _, p_val = stats.wilcoxon(nonzero)
        if p_val < alpha:
            rejections += 1

    return rejections / n_sims


def main():
    # 1. Estimate empirical OR
    empirical_or = estimate_or()

    # 2. Power table
    print("\n" + "=" * 65)
    print("Power Analysis: Pipeline vs Baseline")
    print("  Primary endpoint: Accuracy (single, no Holm correction)")
    print("  ChatGPT comparison: descriptive only (no formal test)")
    print(f"  Baseline distribution: {list(BASELINE_PROBS)}")
    print(f"  Raters: 6 | Alpha: 0.05 (two-sided) | Sims: 5000/cell")
    print("=" * 65)

    ors = [1.3, 1.5, 2.0, 2.5, 3.0]
    ns = [20, 30, 40, 50, 60, 80, 100]

    print(f"\n{'n':>5s}", end="")
    for or_val in ors:
        print(f"  OR={or_val}", end="")
    print()
    print("-" * 55)

    for n in ns:
        print(f"{n:5d}", end="")
        for or_val in ors:
            power = simulate_power(n, n_raters=6, OR=or_val)
            marker = " ✓" if power >= 0.80 else ""
            print(f"  {power:.2f}{marker}", end="")
        print()

    print(f"\n✓ = ≥ 80% power")

    # 3. Summary
    p40 = simulate_power(40, 6, 2.0)
    print(f"\n=== Summary ===")
    print(f"With n=40 scenarios and OR=2.0 (conservative): power = {p40:.0%}")
    print(f"Recommendation: 40 scenarios (20 breast + 20 PDAC) is sufficient")
    print(f"  if the true effect is OR ≥ 2.0")
    print(f"  (LLM judge suggests OR ≈ {empirical_or:.1f}, human raters likely lower)")


if __name__ == '__main__':
    main()
