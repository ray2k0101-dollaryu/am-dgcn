# AM-DGCN — does the multi-scale graph help?

Code and result files for

> **Does the Multi-Scale Graph Help? A Systematic Controlled Study of AM-DGCN for
> EEG Emotion Recognition** — Lei Yu, Rongtao Ding, Decheng Wang —
> *Brain Sciences* (MDPI), 2026.

The paper asks whether a multi-scale graph hierarchy contributes measurable
**cross-subject** benefit to EEG-based emotion recognition, and answers with a
controlled study on SEED, SEED-IV and DEAP.  Its conclusion is largely negative:
the gains are carried by a parallel **linear shortcut**, not by the graph, and a
graph-free variant matches the full model within the across-subject noise floor.
This repository exists so that the negative result can be checked, not taken on
trust.

---

## What "reproduce" means here — two tiers

Reproduction is split in two on purpose, because the honest answer to "can I
re-run this?" is different for each tier.

### Tier 1 — regenerate every table value and every figure (seconds, no dataset)

The result files in `results/` are the complete numerical record of the study.
They are small (a few hundred kilobytes in total) and self-contained, so all
seven figures and every tabulated number can be reproduced **without the
datasets, without a GPU, and without any training**:

```bash
python code/nature_figures.py        # writes figures/*.{pdf,png,svg}
```

This regenerates the figures **byte-identically** to the ones shipped here.  You
can then open `manifest.md` and check any table cell against the JSON that holds
it, entry by entry.

### Tier 2 — re-derive the result files themselves (needs the datasets, hours of CPU)

Re-running the models requires SEED, SEED-IV and DEAP, which are **not** in this
repository (see [Datasets](#datasets)).  With them in place, the run scripts in
`code/` reproduce the result files, and everything above follows.  Rough cost on
CPU: the ablation protocol takes ≈58 s per subject-run, so ten variants × 15
subjects × 2 datasets is ≈4.8 h per seed.

Everything shipped here can be checked in Tier 1.  Tier 2 is what would produce
it from scratch.

---

## Quick start

```bash
pip install -r requirements.txt
python code/nature_figures.py
```

If `figures/` and `results/` sit where they do in this repository, no
configuration is needed.  To point the code somewhere else:

| Variable | Default | Used for |
|---|---|---|
| `AMDGCN_RESULTS` | `./results` | where the checkpoints are read from |
| `AMDGCN_FIGURES` | `./figures` | where the figures are written |
| `AMDGCN_DATA`    | `./data`    | datasets root — Tier 2 only |

## Environment

Produced with Python 3.13.14 on Windows, CPU only.  Figure regeneration needs
only the first two rows; the rest are for retraining.

| Package | Version |
|---|---|
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| torch | 2.6.0+cpu |
| torch-geometric | 2.8.0.post1 |

`mne` and `h5py` are imported defensively (guarded by `HAS_MNE` / `HAS_H5PY`)
and are only needed for part of the preprocessing and for MATLAB v7.3 files.
`pandas` is not used.

## Layout

```
code/                 14 scripts + paths.py
  am_dgcn_model.py        model, adjacency construction, Trainer
  run_experiment.py       data loading, DE features, subject-dependent runs
  run_all_v3.py           main subject-independent driver
  baselines.py            re-implemented DGCNN and RGNN
  run_ablation_*.py       ten-variant ablation (SEED / SEED-IV / multi-seed)
  run_baselines.py        same-protocol baseline re-runs
  run_hp_all.py           hyperparameter sweeps (drives run_hyperparam.py)
  linear_ceiling.py       logistic-regression ceiling on identical features
  paired_test.py          AM-DGCN vs ceiling, paired by held-out subject
  ablation_paired_test.py ablation contrasts against the Full model
  nature_figures.py       all seven figures
  paths.py                the only place that resolves a directory

results/              20 checkpoints + sd_runs/ (4 per-run SD records)
figures/              7 figures x 3 formats
manifest.md           table/figure -> script -> result file
verify_repo.py        checks the claims in manifest.md and README.md
```

`manifest.md` is the file to read if you want to check a specific number: it
maps every table and figure to the script and result key behind it, and it also
warns that **figure file numbers do not match paper figure numbers** (paper
Figure 2 is `figure6_evaluation_protocol`, paper Figure 3 is `figure5_ceiling`,
and so on).

## Datasets

None is redistributed here — neither the raw recordings nor anything derived
from them.

| Dataset | Source | Classes | Subjects |
|---|---|---|---|
| SEED | SJTU BCMI laboratory, <https://bcmi.sjtu.edu.cn/> | 3 | 15 |
| SEED-IV | SJTU BCMI laboratory, <https://bcmi.sjtu.edu.cn/> | 4 | 15 |
| DEAP | Queen Mary University of London, <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/> | 2 | 32 |

SEED and SEED-IV require an application and a signed licence from the SJTU BCMI
laboratory; DEAP likewise has its own terms.  Because those terms do not permit
redistribution, this repository also leaves out the derived feature caches
(`*.npz`, 205 MB) that the training runs build on first use.  Put the datasets
under `./data` as `SEED/`, `SEED_IV/` and `DEAP/`, or set `AMDGCN_DATA`.

Note that the DEAP labels are binarised at a threshold of 5 (`run_experiment.py`),
which makes the majority-class rate 55.31% for valence and 57.58% for arousal.
Those are the numbers to compare the DEAP ceiling against.

## How the released code differs from the authors' working copy

Only in one respect: the working copies hard-coded absolute paths to the
authors' local working directory in 14 places across 7 scripts.  Those are
replaced here by `code/paths.py`, which resolves everything relative to the
repository root with environment overrides.  No algorithmic code was changed,
and the figures regenerate byte-identically, which is the check that this claim
holds.

---

## Reproduction boundaries — please read before quoting a number

These are stated because they limit what the repository can establish, not
because they are incidental.

1. **Figure 3 embeds its values as literals.**  `figure5_ceiling` does not read
   a result file; its bar heights and significance stars are dictionaries in
   `nature_figures.py`.  The same values are independently present in the
   `linear_ceiling_*.json` files and `paired_test_AM-DGCN_vs_ceiling.json`.

2. **Figure 2 has no generator.**  `figure6_evaluation_protocol.svg` is
   hand-authored (the SVG is included as the source) and embeds two literals,
   `n = 15–32` and `≤ 12.4 pp`, that nothing machine-checks.

3. **The subject-dependent linear probe is a split-dependent quantity.**  The
   paper reports it twice, 77.48% and 78.81%, because which of the two lands
   above or below the ceiling depends on the fold split; `m5_sd_split_repro.json`
   is the file behind that.  Neither number should be quoted alone.  The same
   caveat means the DEAP-valence comparison flips sign between splitters.

4. **Ablation axis limits are observed extrema, not round numbers** — 38–88 for
   SEED and 22–72 for SEED-IV.  Rounder limits clip per-subject markers.

5. **The re-implemented baselines are lower bounds, not reproductions.**  DGCNN
   and RGNN were re-run under one common protocol with deliberate
   implementation differences: DGCNN pooling is over the band axis rather than
   the time axis and its adjacency is computed from band profiles; RGNN uses a
   learnable row-stochastic adjacency (distance prior plus kNN, shared across
   the batch) with two `bmm` graph-convolution layers and NodeDAT.  EmotionDL
   was **not** implemented.  The paper's point is that such re-runs shift these
   models by 18–38 points against their published figures and can reverse their
   ranking — so treat these numbers as a floor on what the published
   configurations achieve, not as those configurations.

6. **No trained weights are stored anywhere.**  Reproduction means re-running
   training; nothing in the project ever serialised a model.

## Verification

```bash
python verify_repo.py
```

Checks that the repository contains no absolute path or author-identifying
string, nothing oversized or derived from a licensed dataset, that the figures
regenerate byte-identically from `results/`, and that `manifest.md` accounts for
every table and figure referenced in the paper.  It reports remaining
author-supplied placeholders (there is one: the repository URL) as a count
rather than failing on them.

## Citation and licence

Code and result files: MIT (see `LICENSE`).  Please cite the paper — see
`CITATION.cff`.  The datasets keep their own terms and are not covered by that
licence.
