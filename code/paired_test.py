"""Paired significance test: AM-DGCN SI vs linear-ceiling SI (LOSO).

Pure standard-library Python (no numpy/scipy available in this environment).
Implements:
  * paired Student t-test (two-sided p via regularized incomplete beta)
  * Wilcoxon signed-rank test (EXACT via DP on rank-sum for n<=20,
    normal approximation w/ continuity correction for n>20)

Per-subject SI accuracy is retained for:
  AM-DGCN  : outputs/<...>/si_results.json  (per_subject_accuracy)
  ceiling   : outputs/linear_ceiling.log     (subj k: acc=...)
Paired by subject index under the identical LOSO protocol.
"""
import json, math, re, os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

# ---------- stats (pure stdlib) ----------
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
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    dm = sum(d) / n
    var = sum((x - dm) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return dm, float('nan'), 0.0 if dm == 0 else 0.0
    t = dm / (sd / math.sqrt(n))
    # two-sided p
    x = (n - 1) / ((n - 1) + t * t)
    p = _betai((n - 1) / 2.0, 0.5, x)  # = P(|T| >= |t|)
    return dm, t, p

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def wilcoxon_paired(a, b):
    d = [a[i] - b[i] for i in range(len(a))]
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return None, None
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
        # exact via DP on (scaled) rank-sum
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
        p = 2 * (1 - _norm_cdf(abs(z)))
        method = "normal-approx"
    return W, p, method, n

# ---------- load AM-DGCN per-subject SI (keep only SI runs) ----------
def _load(path):
    d = json.load(open(path))
    if "si_accuracy" in d and "per_subject_accuracy" not in d:
        # sd_results style file: no per-subject SI -> return None
        return None
    return d.get("per_subject_accuracy")

amd = {
    "SEED": _load(os.path.join(OUT, "SEED_all_20260725_211045/si_results.json")),
    "SEED-IV": _load(os.path.join(OUT, "SEED-IV_si_20260726_102125/si_results.json")),
    "DEAP-valence": _load(os.path.join(OUT, "DEAP_si_valence_20260726_113740/si_results.json")),
    "DEAP-arousal": _load(os.path.join(OUT, "deap_arousal_si_result.json")),
}

# ---------- load ceiling per-subject SI (json si_per_subject, log fallback) ----------
ceil_raw = {}
for tag in ["SEED", "SEED-IV", "DEAP-valence", "DEAP-arousal"]:
    fn = os.path.join(OUT, f"linear_ceiling_{tag}.json")
    if os.path.exists(fn):
        d = json.load(open(fn))
        if d.get("si_per_subject"):
            ceil_raw[tag] = d["si_per_subject"]
# log fallback for datasets whose ceiling json predates si_per_subject
cur = None
with open(os.path.join(OUT, "linear_ceiling.log")) as f:
    for line in f:
        s = line.strip()
        if s.startswith("Dataset:") and "15 subjects, 4 classes" in s:
            cur = "SEED-IV"
        elif s.startswith("Found") and "DEAP (valence)" in s:
            cur = "DEAP-valence"
        elif s.startswith("Found") and "DEAP (arousal)" in s:
            cur = "DEAP-arousal"
        elif s.startswith("[SI] linear ceiling") and cur:
            # start of a fresh SI block for the current dataset; reset its list
            # only if json did not already supply per-subject scores
            if cur not in ceil_raw:
                ceil_raw[cur] = []
        elif s.startswith("subj") and cur and cur in ceil_raw:
            m = re.search(r"acc=([0-9.]+)", s)
            if m:
                ceil_raw[cur].append(float(m.group(1)))

# ---------- aggregate ceiling means (from json) ----------
ceil_agg = {}
for tag in ["SEED", "SEED-IV", "DEAP-valence", "DEAP-arousal"]:
    fn = os.path.join(OUT, f"linear_ceiling_{tag}.json")
    if os.path.exists(fn):
        ceil_agg[tag] = json.load(open(fn))["si_accuracy_mean"]

# ---------- run paired tests where BOTH per-subject exist ----------
paired = {}
for tag in ["SEED", "SEED-IV", "DEAP-valence", "DEAP-arousal"]:
    if not amd.get(tag):
        continue
    a = amd[tag]
    b = ceil_raw.get(tag)
    if not b:
        continue  # no per-subject ceiling retained for this dataset
    assert len(a) == len(b), f"{tag}: len mismatch {len(a) if a else 0} vs {len(b) if b else 0}"
    dm, t, p_t = paired_ttest(a, b)
    W, p_w, method, n = wilcoxon_paired(a, b)
    n_zero = sum(1 for x in (a[i] - b[i] for i in range(len(a))) if x == 0)
    paired[tag] = {
        "n_subjects_t": len(a),
        "n_subjects_wilcoxon": n,
        "n_tied": n_zero,
        "amdgc_mean": sum(a) / len(a),
        "ceiling_mean": sum(b) / len(b),
        "mean_diff": dm,
        "paired_t": t,
        "t_df": len(a) - 1,
        "t_pvalue": p_t,
        "wilcoxon_W": W,
        "wilcoxon_pvalue": p_w,
        "wilcoxon_method": method,
    }

# ---------- aggregate-only comparison for any dataset lacking joint per-subject scores ----------
agg_only = {}
for tag in ["SEED", "SEED-IV", "DEAP-valence", "DEAP-arousal"]:
    if tag in paired:
        continue
    if tag in amd and tag in ceil_agg:
        a = amd[tag]
        agg_only[tag] = {
            "n_subjects": len(a),
            "amdgc_mean": sum(a) / len(a),
            "ceiling_mean": ceil_agg[tag],
            "mean_diff": (sum(a) / len(a)) - ceil_agg[tag],
            "note": "ceiling per-subject SI not retained; aggregate comparison only",
        }
    elif tag in ceil_agg:
        agg_only[tag] = {"ceiling_mean": ceil_agg[tag],
                         "note": "AM-DGCN per-subject SI not retained; aggregate ceiling only"}

result = {
    "paired": paired,
    "aggregate_only": agg_only,
    "assumption": "Per-subject SI scores paired by subject index under identical LOSO protocol (AM-DGCN si_results.json vs linear_ceiling.py log).",
}
with open(os.path.join(OUT, "paired_test_AM-DGCN_vs_ceiling.json"), "w") as f:
    json.dump(result, f, indent=2)

# ---------- print ----------
print("=== PAIRED TESTS (both per-subject retained) ===")
for tag, r in paired.items():
    nzero = r.get('n_tied', 0)
    print(f"\n[{tag}]  t-test n={r['n_subjects_t']} (df={r['t_df']}),  Wilcoxon n={r['n_subjects_wilcoxon']} ({nzero} ties excluded)")
    print(f"  AM-DGCN SI mean = {r['amdgc_mean']*100:.2f}%   ceiling SI mean = {r['ceiling_mean']*100:.2f}%")
    print(f"  mean paired diff = {r['mean_diff']*100:+.2f} pp")
    print(f"  paired t = {r['paired_t']:.3f},  p(two-sided) = {r['t_pvalue']:.4g}")
    print(f"  Wilcoxon W = {r['wilcoxon_W']:.1f},  p = {r['wilcoxon_pvalue']:.4g}  [{r['wilcoxon_method']}]")
print("\n=== AGGREGATE-ONLY (per-subject not jointly retained) ===")
for tag, r in agg_only.items():
    print(f"  [{tag}] AM-DGCN {r.get('amdgc_mean',0)*100:.2f}%  vs ceiling {r.get('ceiling_mean',0)*100:.2f}%  (diff {r.get('mean_diff',0)*100:+.2f} pp)")
print("\n[saved] outputs/paired_test_AM-DGCN_vs_ceiling.json")
