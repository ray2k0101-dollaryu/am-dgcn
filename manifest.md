# Manifest: which file produces which number

Every table and figure in the paper, mapped to the script that produces it and
the result file that holds its values.  Nothing here needs to be *run* to be
checked — a reader with the JSON in hand can verify each entry by hand.

---

## 0. Read this first: file names do not match figure numbers

`nature_figures.py` was written incrementally, so the **file** numbers are the
order the code was added, not the order the figures appear in the paper.  The
two orders cross.  Get this wrong once and every downstream check is wrong:

| Paper | File in `figures/` | Submission name |
|---|---|---|
| Figure 1 — Architecture | `figure1_architecture` | `Figure1_Architecture` |
| Figure 2 — Evaluation protocol | **`figure6_evaluation_protocol`** | `Figure2_EvaluationProtocol` |
| Figure 3 — Linear ceiling | **`figure5_ceiling`** | `Figure3_LinearCeiling` |
| Figure 4 — Linear shortcut | **`figure7_shortcut_signal`** | `Figure4_LinearShortcut` |
| Figure 5 — SEED ablation | **`figure2_seed_ablation`** | `Figure5_SEED_Ablation` |
| Figure 6 — SEED-IV ablation | **`figure3_seediv_ablation`** | `Figure6_SEEDIV_Ablation` |
| Figure 7 — Hyperparameters | **`figure4_hyperparams`** | `Figure7_Hyperparameters` |

Table numbers, by contrast, are in order, with `5b`, `6b` and `1b` as paired
extensions (`Table 1b` reports the per-subject pairing behind `Table 1`, and
`Table 5b`/`6b` are same-protocol baseline re-runs).

---

## 1. Tables

| Paper | What it reports | Script | Result file (key) |
|---|---|---|---|
| Table 1 | LOSO linear ceiling per dataset | `code/linear_ceiling.py` | `linear_ceiling_SEED.json`, `-SEED-IV.json`, `-DEAP-valence.json`, `-DEAP-arousal.json` — `sd_accuracy_mean`, `si_accuracy_mean` |
| Table 1b | AM-DGCN SI vs the LOSO ceiling, paired by held-out subject | `code/paired_test.py` | `paired_test_AM-DGCN_vs_ceiling.json` → `paired/<dataset>` (`amdgc_mean`, `ceiling_mean`, `p_t`, `p_wilcoxon`) |
| Table 2 | Subject-dependent SEED | main SD run | `sd_runs/SEED_all_20260725_211045/sd_results.json` |
| Table 3 | Subject-dependent SEED-IV | main SD run | `sd_runs/SEED-IV_all_20260725_214911/sd_results.json` |
| Table 4 | Subject-dependent DEAP (valence / arousal) | main SD run | `sd_runs/DEAP_all_valence_20260726_110845/sd_results.json`, `sd_runs/DEAP_all_arousal_20260726_133255/sd_results.json` |
| Table 5 | Subject-independent SEED (LOSO) | `code/paired_test.py` | `paired_test_AM-DGCN_vs_ceiling.json` → `paired/SEED/amdgc_mean` |
| Table 6 | Subject-independent SEED-IV (LOSO) | `code/paired_test.py` | same file → `paired/SEED-IV/amdgc_mean` |
| Table 7 | Subject-independent DEAP (LOSO) | `code/paired_test.py` | same file → `paired/DEAP-valence`, `paired/DEAP-arousal`; the arousal row is also in `deap_arousal_si_result.json` |
| Table 5b | Same-protocol re-run of DGCNN / RGNN, SEED | `code/run_baselines.py` | `baseline_checkpoint.json` (`SEED/<model>`), `baseline_paired_test.json` |
| Table 6b | Same-protocol re-run, SEED-IV | `code/run_baselines.py` | `baseline_checkpoint.json` (`SEED-IV/<model>`), `baseline_paired_test.json` |
| Table 8 | Ten-variant ablation, SEED | `code/run_ablation_seed.py`, `code/run_ablation_multiseed.py` | `seed_ablation_checkpoint.json`, `seed_ablation_checkpoint_seed20260919.json`, `seed_ablation_checkpoint_seed20260920.json`, `ablation_paired_test.json`, `m3_multiseed_summary.json` |
| Table 9 | Ten-variant ablation, SEED-IV | `code/run_ablation_seediv.py`, `code/run_ablation_multiseed.py` | `seediv_ablation_checkpoint.json`, `seediv_ablation_checkpoint_seed20260919.json`, `seediv_ablation_checkpoint_seed20260920.json`, `ablation_paired_test.json`, `m3_multiseed_summary.json` |
| Table 10 | Fusion-weight `alpha` sweep on SEED | `code/run_hyperparam.py` | `hyperparam_alpha_SEED_checkpoint.json` |
| Table 11 | GCN depth sweep on SEED | `code/run_hyperparam.py` | `hyperparam_layers_SEED_checkpoint.json` |
| Table 12 | Hidden-dimension sweep on SEED | `code/run_hyperparam.py` | `hyperparam_hidden_SEED_checkpoint.json` |
| Table A1 (Appendix A) | Model complexity in parameters | the three sweeps above | `n_params` inside each `hyperparam_*_SEED_checkpoint.json` |

### Two numbers that are a free parameter of the data split

The paper's Table 2 note reports the subject-dependent linear probe twice,
**77.48%** and **78.81%**, and explains that the difference comes from the fold
split rather than from the probe.  `results/m5_sd_split_repro.json` is the file
behind that claim: it replays the same probe under three splitters (the
published `StratifiedKFold(5, shuffle, seed 0)`, the same family with seed 42,
and the `KFold(5, shuffle, 42)` used by the main experiments) and records which
of the two the resulting ceiling falls above or below.  Both figures are
reported in the paper on purpose; neither should be quoted alone.

### Result files not shipped

`outputs/` on the machine that produced the paper holds 24 JSON checkpoints.
20 are here, plus the 4 per-run `sd_results.json` records above.  The ones held
back, with reasons:

| File | Why it is not here |
|---|---|
| `all_remaining_results.json` | contains an **earlier** SEED ablation batch whose `ablation` block predates the `Single Scale (GCN+Linear)` variant, i.e. its ablation numbers are not the published ones. Superseded by `seed_ablation_checkpoint.json`. |
| `baseline_checkpoint_smoke.json`, `..._smokeM.json`, `..._smokergnn.json` | single-model smoke tests, no reported value |
| six further `sd_results.json` under other run folders (SEED 0.4978, SEED 0.6133, `seed_fixed_sdsi` 0.8311, `DEAP_sd_valence` 0.5563, …) | earlier or aborted runs, none cited in the paper. Shipping them next to the published 83.85 for "SEED SD" would invite a misreading. |

Derived feature caches (`*.npz`, 205 MB) and run logs are also excluded — see
`README.md` for why.

---

## 2. Figures

All figures are written to `figures/` in three formats (`.pdf`, `.png`, `.svg`)
by `code/nature_figures.py`, **except Figure 2**, and all of them are
regenerated by a single command:

```bash
python code/nature_figures.py
```

| Paper | Generated by | Values come from |
|---|---|---|
| Figure 1 — Architecture | `nature_figures.py` | nothing: a schematic drawn with matplotlib boxes and arrows |
| **Figure 2 — Evaluation protocol** | **nothing in this repository** | see below |
| Figure 3 — Linear ceiling | `nature_figures.py` | **literals inside the script**, not a JSON — see below |
| Figure 4 — Linear shortcut | `nature_figures.py` | `ablation_paired_test.json`, `baseline_checkpoint.json`, `baseline_paired_test.json`, `seed_ablation_checkpoint.json`, `seediv_ablation_checkpoint.json`, `linear_ceiling_SEED.json`, `linear_ceiling_SEED-IV.json` |
| Figure 5 — SEED ablation | `nature_figures.py` | `seed_ablation_checkpoint.json` (`per_subject_accuracy`; 150 points drawn) |
| Figure 6 — SEED-IV ablation | `nature_figures.py` | `seediv_ablation_checkpoint.json` (150 points drawn) |
| Figure 7 — Hyperparameters | `nature_figures.py` | `hyperparam_alpha_SEED_checkpoint.json`, `hyperparam_layers_SEED_checkpoint.json`, `hyperparam_hidden_SEED_checkpoint.json` |

### Figure 2 is an authored diagram, not a plot

`figures/figure6_evaluation_protocol.svg` is hand-authored SVG (custom CSS
classes, no matplotlib metadata).  There is no script that produces it, because
there is nothing data-driven in it: it is a flow diagram of the evaluation
protocol.  **The SVG is the source**, and it is included, so the figure is
editable and verifiable — but it is not *generated*.

It does embed two literals that a future revision must keep in sync:

- `n = 15–32` — the subject counts across the three datasets
- `≤ 12.4 pp` — the bounded advantage over the ceiling, which is also the
  `4.8–12.4` range in the abstract

Nothing machine-checks these two strings, so treat this file as a place where a
number can go stale.

### Figure 3 embeds its values as literals

`figure5_ceiling` (paper Figure 3) does **not** read a result file.  Its eight
bar heights and the significance stars are dictionaries written directly into
`nature_figures.py`.  The same numbers are independently present in
`linear_ceiling_SEED.json`, `linear_ceiling_SEED-IV.json`,
`linear_ceiling_DEAP-valence.json`, `linear_ceiling_DEAP-arousal.json` and
`paired_test_AM-DGCN_vs_ceiling.json`, which is what makes the literals
checkable — but the script itself is the source of truth for the drawn figure,
and editing one side has no effect on the other.

### Ablation axis limits are not round numbers on purpose

`figure2_seed_ablation` (Figure 5) uses a y-range of 38–88 and
`figure3_seediv_ablation` (Figure 6) uses 22–72.  These are the observed extrema
of the per-subject accuracies, not tidy limits: seven of the ten SEED variants
have a subject above 80 (maximum 84.44) and two SEED-IV variants have one below
28 (minimum 23.61).  Rounder limits truncate those markers, so do not "tidy"
them without re-checking `per_subject_accuracy`.

---

## 3. Reproducing all of it

```bash
python code/nature_figures.py            # 7 figures -> figures/ (byte-identical)
```

That is the whole first tier: the shipped result files are sufficient.  No
dataset, no GPU, no training.  For the second tier — re-deriving the result
files themselves — see `README.md`, which documents the datasets and the run
scripts, and `verify_repo.py`, which checks the claims made in this file.
