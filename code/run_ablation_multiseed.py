"""Multi-seed repeat of the 10-variant ablation (SEED and SEED-IV).

Why this exists
---------------
The pre-submission review asked for a multi-seed repeat: the paper reports a
single run per variant, and Section 4.6 itself records that the sign of several
component effects flips between its two ablation batches. Mean and standard
deviation over independent seeds is the direct answer.

Design
------
* One seed is applied at module level, then SEED runs, then SEED-IV runs -- the
  exact structural shape of run_ablation_fresh_all.py, so the seed-20260918
  round is directly comparable to (and is reused from) the authoritative
  2026-09-18 checkpoint.
* Each seed gets its own file namespace via AMDGCN_RUN_TAG (see run_all_v3.py
  and run_ablation_seediv.py). With a tag set, the run can neither overwrite
  nor accidentally resume into the authoritative checkpoints.
* The seed is re-applied at the start of every round, so rounds are independent
  rather than continuing one RNG stream.
* Model code is NOT modified by this runner.

Seeds
-----
  20260918  reused from outputs/*_ablation_checkpoint.json (not re-run here)
  20260919  run here
  20260920  run here

Set AMDGCN_SEEDS to override the list, e.g. AMDGCN_SEEDS=20260919,20260920,20260921

Usage (foreground PowerShell, torch on CPU):
    $env:PYTHONIOENCODING='utf-8'
    & $py -X utf8 run_ablation_multiseed.py

Logging is file-only plus the real stdout when one exists, because this may be
launched without an attached console.
"""
import io
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "ablation_multiseed.log")

_fh = io.open(LOG, "a", encoding="utf-8")


class _Sink(object):
    """Write to the log file, and to the real stdout only if one exists."""

    def __init__(self, fh):
        self.fh = fh
        self.real = sys.__stdout__

    def write(self, data):
        try:
            self.fh.write(data)
            self.fh.flush()
        except Exception:
            pass
        if self.real is not None:
            try:
                self.real.write(data)
                self.real.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.fh.flush()
        except Exception:
            pass


sys.stdout = _Sink(_fh)
sys.stderr = sys.stdout

SEEDS = [int(s) for s in os.environ.get("AMDGCN_SEEDS", "20260919,20260920").split(",") if s.strip()]
WATCH = [
    "Full AM-DGCN",
    "w/o Linear Shortcut",
    "Single Scale (GCN)",
    "Single Scale (GCN+Linear)",
    "Linear Only (no GCN)",
]


def _reseed(seed):
    """Re-apply one seed to every RNG that the training loop consumes."""
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception as exc:  # pragma: no cover
        print("[multiseed] numpy seed skipped: %r" % (exc,))
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
    except Exception as exc:  # pragma: no cover
        print("[multiseed] torch seed skipped: %r" % (exc,))


print("=" * 74)
print("[multiseed] START %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
print("[multiseed] cwd       %s" % HERE)
print("[multiseed] log       %s" % LOG)
print("[multiseed] new seeds %s" % SEEDS)
print("[multiseed] reused    seed 20260918 from outputs/*_ablation_checkpoint.json")
print("=" * 74)

for seed in SEEDS:
    tag = "_seed%d" % seed
    os.environ["AMDGCN_RUN_TAG"] = tag
    _reseed(seed)

    print("\n" + "#" * 74)
    print("[multiseed] ROUND seed=%d  tag=%s  %s" % (seed, tag, time.strftime("%Y-%m-%d %H:%M:%S")))
    print("#" * 74)

    print("\n[multiseed] ---------------- SEED ----------------")
    import run_ablation_seed
    run_ablation_seed.main()

    print("\n[multiseed] ---------------- SEED-IV ----------------")
    import run_ablation_seediv
    run_ablation_seediv.main()

    # Quick eyeball summary for this round.
    import json
    for name, path in (("SEED", os.path.join(OUT, "seed_ablation_checkpoint%s.json" % tag)),
                       ("SEED-IV", os.path.join(OUT, "seediv_ablation_checkpoint%s.json" % tag))):
        if not os.path.exists(path):
            print("[multiseed] !! missing %s" % path)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        print("[multiseed] %s seed=%d  (%d/%d variants)" % (name, seed, len(res), 10))
        for v in WATCH:
            if v in res:
                print("    %-28s %.2f%%" % (v, res[v]["mean_accuracy"] * 100))

print("\n" + "=" * 74)
print("[multiseed] DONE %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 74)
