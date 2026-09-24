"""Same-protocol LOSO runner for the baseline GNN models (review item A2).

The protocol is copied verbatim from the AM-DGCN ablation runner
(run_ablation_seediv.py) so that the comparison is genuinely matched:

    LOSO over held-out subjects; per fold, train on the other 14 subjects.
    Adam(lr=1e-3, weight_decay=1e-4); batch 64; <=60 epochs;
    early stopping on the held-out fold with patience 20;
    ReduceLROnPlateau(factor=0.5, patience=10) stepping on the same metric;
    best per-subject accuracy and macro-F1 retained, i.e. the number recorded
    is the best epoch of that fold.

The early stopping metric is the held-out fold itself, exactly as in the
existing ablation and headline runs.  That is a known weakness of the protocol
(the manuscript states it in Section 5.7 as test-set reuse), but it is what the
AM-DGCN numbers were produced under, so the baselines must use it too or the
comparison would be invalid.  Do not "fix" it here alone.

Per-subject arrays are retained (outputs/baseline_checkpoint.json) so that the
paired t / Wilcoxon tests used in Section 4.6 can be applied to the baselines as
well.

Crash-resilient: per-model per-subject progress is flushed to
outputs/baseline_tmp/<dataset>__<model>.json, and a re-launch resumes.
"""
import os
import sys
import json
import gc
import traceback
from paths import dataset_dir

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
os.chdir(_script_dir)

# A smoke test must not leave per-subject progress behind, or the real batch
# would "resume" from it and mix protocols.  BASELINE_SUFFIX isolates the
# checkpoint and the progress files without touching the code path.
_SUFFIX = os.environ.get("BASELINE_SUFFIX", "").strip()
_SFX = ("_" + _SUFFIX) if _SUFFIX else ""

CKPT = os.path.join("outputs", "baseline_checkpoint%s.json" % _SFX)
TMP_DIR = os.path.join("outputs", "baseline_tmp%s" % _SFX)

DATA_DIRS = {
    "SEED": dataset_dir('SEED'),
    "SEED-IV": dataset_dir('SEED_IV'),
}


def _safe(name):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def model_registry():
    """name -> spec.

    trainer        : Training loop to use.  RGNN needs TrainerRGNN because the
                     NodeDAT term consumes the subject ids; everything else
                     uses the shared Trainer unchanged.
    needs_subject  : whether the training loader must carry 'subject_id'.
    extra          : kwargs merged into the model constructor.
    """
    from baselines import DGCNN, RGNN, TrainerRGNN
    from am_dgcn_model import Trainer
    return {
        "DGCNN": {
            "factory": DGCNN,
            "desc": "published widths, not parameter-matched",
            "trainer": Trainer,
            "needs_subject": False,
            "extra": {},
        },
        # Capacity control.  The published DGCNN carries 201,389 parameters, 3x
        # AM-DGCN's 67,582, so a reviewer can ask whether "no significant
        # difference" is just a well-fed baseline.  This variant rescales the
        # published pyramid (w, 2w, 4w, w) by taking w=36 instead of w=64 --
        # one scalar, nothing else touched -- which lands on 68,249 params
        # (+0.99% vs AM-DGCN).  Same kernels, paddings, adjacency and pooling;
        # only the width of every block changes.
        "DGCNN-matched": {
            "factory": DGCNN,
            "desc": ("capacity-matched to AM-DGCN: published pyramid w=36 "
                     "(68,249 params vs 67,582)"),
            "trainer": Trainer,
            "needs_subject": False,
            "extra": {"conv_channels": (36, 72, 144, 36)},
        },
        "RGNN": {
            "factory": RGNN,
            "desc": "graph backbone + NodeDAT domain-adversarial regulariser",
            "trainer": TrainerRGNN,
            "needs_subject": True,
            "extra": {"use_nodedat": True},
        },
        "RGNN-plain": {
            "factory": RGNN,
            "desc": "graph backbone only, NodeDAT disabled",
            "trainer": Trainer,
            "needs_subject": False,
            "extra": {"use_nodedat": False},
        },
    }


def _make_subject_dataset(torch_mod, base, indices, subject_ids, remap):
    """Wrap the shared dataset so each item also carries its subject id.

    The shared collate_fn is NOT modified; the runner adds the extra key in its
    own collate wrapper below.  The shared dataset class is not modified either
    -- this is a view over it.
    """
    class _DS(torch_mod.utils.data.Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, i):
            j = indices[i]
            d = dict(base[j])
            d["subject_id"] = torch_mod.tensor(
                remap.get(subject_ids[j], -1), dtype=torch_mod.long)
            return d

    return _DS()


def run_baselines(dataset_name="SEED", model_names=("DGCNN",), epochs=60,
                  patience=20, max_subjects=None, hidden_dim=64):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from run_experiment import load_dataset, collate_fn
    from am_dgcn_model import Trainer

    data_dir = DATA_DIRS[dataset_name]
    reg = model_registry()

    ds = load_dataset(dataset_name, data_dir)
    print(ds.summary())
    n_ch = ds.features[0].shape[0]
    n_cls = ds.n_classes
    unique_subjects = sorted(set(ds.subject_ids))
    if max_subjects:
        unique_subjects = unique_subjects[:max_subjects]
    n_total = len(unique_subjects)

    os.makedirs(TMP_DIR, exist_ok=True)

    # checkpoint (resume across launches)
    results = {}
    if os.path.exists(CKPT):
        try:
            with open(CKPT, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            print("[resume] checkpoint unreadable (%r); starting fresh" % (e,))
            results = {}

    for mname in model_names:
        if mname not in reg:
            print("[SKIP] unknown model %r" % (mname,))
            continue
        spec = reg[mname]
        factory = spec["factory"]
        desc = spec["desc"]
        trainer_cls = spec["trainer"]
        extra = dict(spec["extra"])
        key = "%s/%s" % (dataset_name, mname)

        tmp_path = os.path.join(TMP_DIR, _safe(key) + ".json")
        accs, f1s = [], []
        done = 0
        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    _p = json.load(f)
                accs = [float(x) for x in _p.get("accs", [])]
                f1s = [float(x) for x in _p.get("f1s", [])]
                done = len(accs)
                print("  [resume] %s: %d/%d subjects already done" % (key, done, n_total))
            except Exception as e:
                print("  [resume] failed (%r); restarting model" % (e,))
                accs, f1s = [], []

        print("\n--- %s (%s) ---" % (mname, desc))

        def _collate(batch, _base=collate_fn, _need=spec["needs_subject"]):
            """Shared collate, plus subject ids only for models that use them."""
            out = _base(batch)
            if _need:
                out["subject_id"] = torch.stack([b["subject_id"] for b in batch])
            return out

        for idx, subj in enumerate(unique_subjects):
            if idx < done:
                print("  Test Subject %d/%d (id=%s) [skipped]" % (idx + 1, n_total, subj))
                continue
            print("  Test Subject %d/%d (id=%s)" % (idx + 1, n_total, subj))

            train_idx = [i for i, s in enumerate(ds.subject_ids) if s != subj]
            test_idx = [i for i, s in enumerate(ds.subject_ids) if s == subj]

            # NodeDAT needs the subject label of every TRAINING sample, remapped
            # to 0..S-1 because the held-out subject contributes no training data
            # (so the domain classifier must not reserve an output for it).
            train_subjects = sorted(set(ds.subject_ids[i] for i in train_idx))
            remap = {s: k for k, s in enumerate(train_subjects)}

            if spec["needs_subject"]:
                train_ds = _make_subject_dataset(torch, ds, train_idx,
                                                 ds.subject_ids, remap)
            else:
                train_ds = torch.utils.data.Subset(ds, train_idx)

            train_ld = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  collate_fn=_collate)
            test_ld = DataLoader(torch.utils.data.Subset(ds, test_idx),
                                 batch_size=64, shuffle=False, collate_fn=collate_fn)

            build_kwargs = dict(n_channels=n_ch, n_regions=5, n_classes=n_cls,
                                n_bands=5, hidden_dim=hidden_dim, dropout=0.5,
                                n_domain_subjects=len(train_subjects), **extra)
            model = factory(**build_kwargs).to("cpu")
            trainer = trainer_cls(model, device="cpu")
            opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=10)

            best_acc, best_f1, pc = 0.0, 0.0, 0
            for ep in range(epochs):
                # drives the NodeDAT lambda ramp; no-op for models without it
                if hasattr(model, "set_epoch"):
                    model.set_epoch(ep)
                trainer.train_epoch(train_ld, opt, crit)
                vm = trainer.evaluate(test_ld, crit)
                sched.step(vm["accuracy"])
                if vm["accuracy"] > best_acc:
                    best_acc = vm["accuracy"]
                    best_f1 = vm["f1_score"]
                    pc = 0
                else:
                    pc += 1
                if pc >= patience:
                    print("    Early stopping at epoch %d" % (ep + 1))
                    break

            accs.append(float(best_acc))
            f1s.append(float(best_f1))
            print("    Subject %s: Acc=%.4f, F1=%.4f" % (subj, best_acc, best_f1))

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"accs": accs, "f1s": f1s}, f)

            del model, trainer, opt
            gc.collect()

        n_par = sum(p.numel() for p in factory(
            n_channels=n_ch, n_regions=5, n_classes=n_cls, n_bands=5,
            hidden_dim=hidden_dim, dropout=0.5,
            n_domain_subjects=max(1, n_total - 1), **extra).parameters())
        mean = sum(accs) / len(accs)
        # Population sd (ddof=0), matching np.std() in run_ablation_seediv.py and
        # the linear-ceiling script -- i.e. matching what Tables 1, 8 and 9
        # report (SEED Full is 64.89 +/- 10.08).  An earlier version of this
        # runner used ddof=1, which inflated every baseline sd by
        # sqrt(15/14) = 3.5% (DGCNN 11.02 instead of 10.64) and would have put a
        # table next to Table 8 that disagreed with it about the same quantity.
        # If this convention ever changes, normalize the existing checkpoint with
        # tmp/normalise_baseline_sd.py rather than editing numbers by hand.
        std = (sum((a - mean) ** 2 for a in accs) / max(len(accs), 1)) ** 0.5
        results[key] = {
            "model": mname,
            "description": desc,
            "protocol": ("LOSO, %d epochs max, patience %d, batch 64, Adam lr 1e-3"
                         % (epochs, patience)),
            "n_params": n_par,
            "epochs_max": epochs,
            "patience": patience,
            "hidden_dim": hidden_dim,
            "extra_kwargs": extra,
            "nodedat_ramp": extra.get("nodedat_ramp", 20) if extra else None,
            "per_subject_accuracy": accs,
            "per_subject_f1": f1s,
            "mean_accuracy": mean,
            "std_accuracy": std,
            "sd_ddof": 0,
            "mean_f1": sum(f1s) / len(f1s),
            "n_subjects": len(accs),
        }
        with open(CKPT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("  %s: Acc=%.4f +/- %.4f  (params=%d)"
              % (mname, mean, std, n_par))
        print("  [SAVED] Checkpoint updated")

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SEED", choices=["SEED", "SEED-IV"])
    ap.add_argument("--models", default="DGCNN")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--max-subjects", type=int, default=None)
    a = ap.parse_args()
    run_baselines(dataset_name=a.dataset,
                  model_names=tuple(m.strip() for m in a.models.split(",") if m.strip()),
                  epochs=a.epochs, patience=a.patience,
                  max_subjects=a.max_subjects)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
