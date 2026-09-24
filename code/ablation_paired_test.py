"""Paired significance tests ACROSS ablation variants (B1 deliverable).

The existing paired_test.py compares AM-DGCN vs the linear ceiling (Table 1b).
This script instead pairs ablation variants against each other by held-out
subject, using the per-subject accuracy arrays saved by the fresh single-seed
re-run (run_ablation_fresh_all.py, seed=20260918).

Every variant in a dataset is evaluated on the SAME 15 held-out subjects under
the SAME LOSO split, so the per-subject accuracies are naturally paired and a
paired test is valid.

Outputs
-------
  outputs/ablation_paired_test.json   full result (all variants vs Full)
  stdout                              human-readable table

Statistics are implemented in pure stdlib, copied from paired_test.py so that
the numbers stay comparable with the published Table 1b:

  * paired Student t-test, two-sided p via regularized incomplete beta
  * Wilcoxon signed-rank: EXACT via DP for n<=20, normal approx otherwise
  * Cohen's d_z = mean(diff) / sd(diff)          (paired effect size)
  * 95% CI for the mean paired difference (t-based, df = n-1)

Usage:  python ablation_paired_test.py
"""
import json, math, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

DATASETS = [
    ("SEED",    "seed_ablation_checkpoint.json"),
    ("SEED-IV", "seediv_ablation_checkpoint.json"),
]

# The comparisons the reviewer report specifically asks for, plus the pairs the
# prose needs: graph-free vs the minimal GCN + shortcut configuration, and the
# marginal value of the shortcut on top of a bare single-scale GCN.  The last of
# these is the contrast Section 5.1 argues about, and it was missing from the
# original four -- the sentence in the manuscript quoted p = 0.013 for it, which
# is in fact the p-value of the different (Full vs Single Scale GCN) contrast.
REQUIRED = [
    ("Full AM-DGCN", "Single Scale (GCN+Linear)"),
    ("Full AM-DGCN", "Linear Only (no GCN)"),
    ("Full AM-DGCN", "w/o Linear Shortcut"),
    ("Single Scale (GCN+Linear)", "Linear Only (no GCN)"),
    ("Single Scale (GCN+Linear)", "Single Scale (GCN)"),
]


# --------------------------------------------------------------------------
# statistics (pure stdlib; identical implementation to paired_test.py)
# --------------------------------------------------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest(a, b):
    """Two-sided paired Student t-test. Returns (mean_diff, t, p, sd_diff)."""
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    dm = sum(d) / n
    var = sum((x - dm) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return dm, float("nan"), 1.0, 0.0
    t = dm / (sd / math.sqrt(n))
    x = (n - 1) / ((n - 1) + t * t)
    p = _betai((n - 1) / 2.0, 0.5, x)  # P(|T| >= |t|)
    return dm, t, p, sd


def t_crit_two_sided(df, conf=0.95):
    """t critical value via bisection on the two-sided tail probability."""
    alpha = 1.0 - conf
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        x = df / (df + mid * mid)
        tail = _betai(df / 2.0, 0.5, x)          # P(|T| >= mid)
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank. Returns (W, p, method, n_used, n_tied)."""
    d = [a[i] - b[i] for i in range(len(a))]
    n_tied = sum(1 for x in d if x == 0)
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return None, None, "all-tied", 0, n_tied
    absd = [abs(x) for x in d]
    order = sorted(range(n), key=lambda i: absd[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[order[j + 1]] == absd[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_pos = sum(ranks[k] for k in range(n) if d[k] > 0)
    w_neg = sum(ranks[k] for k in range(n) if d[k] < 0)
    W = min(w_pos, w_neg)
    if n <= 20:
        sr = [int(round(2 * r)) for r in ranks]
        total = sum(sr)
        dp = [0] * (total + 1)
        dp[0] = 1
        for r in sr:
            ndp = [0] * (total + 1)
            for s in range(total + 1):
                if dp[s]:
                    ndp[s] += dp[s]
                    if s + r <= total:
                        ndp[s + r] += dp[s]
            dp = ndp
        thresh = min(int(round(2 * w_pos)), total - int(round(2 * w_pos)))
        p = sum(dp[s] for s in range(total + 1)
                if min(s, total - s) <= thresh) / (2 ** n)
        method = "exact"
    else:
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if w_pos > mu:
            z = (w_pos - mu - 0.5) / sigma
        else:
            z = (w_pos - mu + 0.5) / sigma
        p = 2 * (1 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
        method = "normal-approx"
    return W, p, method, n, n_tied


def compare(arr_a, arr_b):
    """Full paired comparison of two per-subject arrays."""
    n = len(arr_a)
    dm, t, p_t, sd = paired_ttest(arr_a, arr_b)
    W, p_w, method, n_w, n_tied = wilcoxon_paired(arr_a, arr_b)
    dz = (dm / sd) if sd > 0 else float("nan")
    tc = t_crit_two_sided(n - 1)
    se = sd / math.sqrt(n)
    ci_lo, ci_hi = dm - tc * se, dm + tc * se
    return {
        "n": n,
        "mean_a": sum(arr_a) / n,
        "mean_b": sum(arr_b) / n,
        "mean_diff": dm,
        "sd_diff": sd,
        "paired_t": t,
        "t_df": n - 1,
        "t_pvalue": p_t,
        "cohens_dz": dz,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "t_crit": tc,
        "wilcoxon_W": W,
        "wilcoxon_pvalue": p_w,
        "wilcoxon_method": method,
        "wilcoxon_n": n_w,
        "n_tied": n_tied,
        "significant_05": (p_t is not None and p_t < 0.05),
    }


def fmt_p(p):
    if p is None:
        return "  n/a  "
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def main():
    report = {}
    for ds_name, fname in DATASETS:
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            print(f"[skip] {ds_name}: {fname} not found")
            continue
        d = json.load(open(path, encoding="utf-8"))
        per_subj = {}
        for variant, v in d.items():
            ps = v.get("per_subject_accuracy")
            if ps:
                per_subj[variant] = [float(x) for x in ps]

        if "Full AM-DGCN" not in per_subj:
            print(f"[skip] {ds_name}: no 'Full AM-DGCN' with per-subject array")
            continue

        ref = per_subj["Full AM-DGCN"]
        bad = [k for k, v in per_subj.items() if len(v) != len(ref)]
        if bad:
            print(f"[warn] {ds_name}: length mismatch for {bad}; dropped")
            for k in bad:
                per_subj.pop(k)
            ref = per_subj["Full AM-DGCN"]

        print("=" * 108)
        print(f"{ds_name}  —  {len(per_subj)} variants, n = {len(ref)} held-out subjects")
        print("=" * 108)
        print(f"{'comparison (A vs B)':<46} {'mean A':>8} {'mean B':>8} {'diff':>7} "
              f"{'95% CI':>17} {'t':>7} {'p(t)':>8} {'dz':>6} {'p(W)':>8}")
        print("-" * 108)

        rows = {}
        for variant in per_subj:
            if variant == "Full AM-DGCN":
                continue
            r = compare(ref, per_subj[variant])
            rows[variant] = r
            ci = f"[{r['ci95_low']*100:+.2f},{r['ci95_high']*100:+.2f}]"
            print(f"{'Full AM-DGCN vs ' + variant:<46} "
                  f"{r['mean_a']*100:>7.2f}% {r['mean_b']*100:>7.2f}% "
                  f"{r['mean_diff']*100:>+6.2f} {ci:>17} "
                  f"{r['paired_t']:>7.3f} {fmt_p(r['t_pvalue']):>8} "
                  f"{r['cohens_dz']:>6.2f} {fmt_p(r['wilcoxon_pvalue']):>8}")

        # ---- effect-size ranking (largest |dz| first) ----
        print("-" * 108)
        print("Ranked by |Cohen's dz| (largest effect first):")
        for variant, r in sorted(rows.items(), key=lambda kv: -abs(kv[1]["cohens_dz"])):
            mag = ("large" if abs(r["cohens_dz"]) >= 0.8 else
                   "medium" if abs(r["cohens_dz"]) >= 0.5 else
                   "small" if abs(r["cohens_dz"]) >= 0.2 else "negligible")
            sig = "SIG" if r["significant_05"] else "n.s."
            print(f"  {variant:<34} dz = {r['cohens_dz']:+.2f} ({mag:<10}) "
                  f"diff = {r['mean_diff']*100:+.2f} pp   {sig} (p_t = {fmt_p(r['t_pvalue'])})")

        # ---- the three required comparisons, explicit ----
        print("-" * 108)
        print("Required comparisons from the reviewer report:")
        for a, b in REQUIRED:
            if a in per_subj and b in per_subj:
                r = compare(per_subj[a], per_subj[b])
                print(f"  {a} vs {b}")
                print(f"     diff = {r['mean_diff']*100:+.2f} pp  "
                      f"95% CI [{r['ci95_low']*100:+.2f}, {r['ci95_high']*100:+.2f}]  "
                      f"t({r['t_df']}) = {r['paired_t']:.3f}  p = {fmt_p(r['t_pvalue'])}  "
                      f"dz = {r['cohens_dz']:+.2f}  "
                      f"Wilcoxon W = {r['wilcoxon_W']} p = {fmt_p(r['wilcoxon_pvalue'])} "
                      f"[{r['wilcoxon_method']}]")
                report.setdefault(ds_name, {}).setdefault("required", {})[
                    f"{a} vs {b}"] = r
            else:
                print(f"  [missing] {a} vs {b}")

        report.setdefault(ds_name, {})["n_subjects"] = len(ref)
        report.setdefault(ds_name, {})["n_variants"] = len(per_subj)
        report.setdefault(ds_name, {})["vs_full"] = rows
        print()

    # ---- cross-dataset summary of the required comparisons ----
    print("=" * 108)
    print("Cross-dataset summary (mean paired diff, pp)")
    print("=" * 108)
    labels = [f"{a.replace('AM-DGCN ','')} vs {b}" for a, b in REQUIRED]
    print(f"{'dataset':<10} " + " ".join(f"{lb[:34]:>36}" for lb in labels))
    for ds in report:
        cells = []
        for a, b in REQUIRED:
            r = report[ds].get("required", {}).get(f"{a} vs {b}")
            cells.append(f"{r['mean_diff']*100:>+10.2f} (p={fmt_p(r['t_pvalue']).strip()})"
                         if r else " " * 36)
        print(f"{ds:<10} " + " ".join(f"{c:>36}" for c in cells))
    print()

    out_path = os.path.join(OUT, "ablation_paired_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
