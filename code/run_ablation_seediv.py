"""
SEED-IV 9-variant ablation runner (fixed-code, crash-resilient, checkpoint resume).

Mirrors run_all_v3.run_ablation_optimized but for the SEED-IV 4-class dataset,
so the component-attribution conclusions can be compared against the SEED
3-class ablation under identical settings.

Checkpoint/resume design (same as the SEED ablation that already completed):
  * outputs/seediv_ablation_checkpoint.json  -> per-variant results (9 keys)
  * outputs/seediv_ablation_tmp/<variant>.json -> per-subject progress for the
    variant currently in flight, so a crash only loses in-progress subjects.
A re-launch of main() resumes from the last completed variant/subject.

Launch (must be via scheduled task / detached, torch long-run):
  python -X utf8 run_ablation_seediv.py
"""
import os
import sys
import json
import time
import gc
import traceback
from paths import dataset_dir

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
os.chdir(_script_dir)

ABL_CKPT = os.path.join('outputs', 'seediv_ablation_checkpoint.json')
TMP_DIR = os.path.join('outputs', 'seediv_ablation_tmp')
DATA_DIR = dataset_dir('SEED_IV')


def run_tag():
    """Optional suffix so a multi-seed repeat gets its own files.

    Empty by default, in which case every path is byte-identical to the
    single-seed behaviour of the 2026-09-18 ablation. A non-empty tag (e.g.
    '_seed20260919') gives that round its own checkpoint and scratch directory.
    """
    return os.environ.get('AMDGCN_RUN_TAG', '')


def tagged(base, tag=None):
    tag = run_tag() if tag is None else tag
    root, ext = os.path.splitext(base)
    return root + tag + ext


def run_ablation_optimized(data_dir, epochs=60, patience=20):
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from run_experiment import load_dataset, collate_fn
    from am_dgcn_model import AMDGCN, Trainer

    abl_ckpt = tagged(ABL_CKPT)
    tmp_dir = TMP_DIR + run_tag()
    print(f"[paths] checkpoint = {abl_ckpt}")
    print(f"[paths] scratch    = {tmp_dir}")
    print(f"[paths] run tag    = {run_tag()!r}")

    ds = load_dataset('SEED-IV', data_dir)
    print(ds.summary())
    n_ch = ds.features[0].shape[0]
    n_cls = ds.n_classes
    unique_subjects = sorted(set(ds.subject_ids))
    n_total = len(unique_subjects)

    # The 9 variants (identical to SEED ablation for a fair comparison).
    variants = {
        'Full AM-DGCN': {},
        'w/o Multi-Scale': {'use_multi_scale': False},
        'w/o Adaptive Adjacency': {'use_adaptive_adj': False},
        'w/o PLV': {'use_plv': False},
        'w/o Cross-Scale MP': {'use_cross_scale': False},
        'w/o Scale Attention': {'use_scale_attention': False},
        'w/o Linear Shortcut': {'use_linear_shortcut': False},
        # Minimal GNN + linear shortcut (all graph-level components off but the
        # linear path kept ON). Comparing against 'Single Scale (GCN)' isolates
        # the marginal value of the linear shortcut.
        'Single Scale (GCN+Linear)': {
            'use_multi_scale': False, 'use_adaptive_adj': False,
            'use_plv': False, 'use_cross_scale': False,
            'use_scale_attention': False, 'use_linear_shortcut': True,
        },
        # Minimal GNN, no linear shortcut (pure electrode-level GCN baseline).
        'Single Scale (GCN)': {
            'use_multi_scale': False, 'use_adaptive_adj': False,
            'use_plv': False, 'use_cross_scale': False,
            'use_scale_attention': False, 'use_linear_shortcut': False,
        },
        # Pure linear shortcut, NO GCN pathway at all (see SEED ablation comment).
        'Linear Only (no GCN)': {
            'use_gcn_pathway': False, 'use_linear_shortcut': True,
        },
    }

    # Load checkpoint (per-variant results).
    results = {}
    if os.path.exists(abl_ckpt):
        with open(abl_ckpt, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"[RESUME] Ablation checkpoint: {len(results)}/{len(variants)} variants done")
        for name in results:
            print(f"  [OK] {name}: Acc={results[name]['mean_accuracy']:.4f}")

    os.makedirs(tmp_dir, exist_ok=True)

    def _safe(n):
        return (n.replace(' ', '_').replace('(', '').replace(')', '')
                 .replace('/', '_').replace('\\', '_'))

    # Run remaining variants.
    for name, variant_kwargs in variants.items():
        if name in results:
            print(f"\n--- {name}: SKIP (already done) ---")
            continue

        print(f"\n--- {name} ---")
        tmp_path = os.path.join(tmp_dir, _safe(name) + '.json')
        accs, f1s = [], []
        done = 0
        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    _p = json.load(f)
                accs = [float(x) for x in _p.get('accs', [])]
                f1s = [float(x) for x in _p.get('f1s', [])]
                done = len(accs)
                print(f"  [resume] {done}/{n_total} subjects already done")
            except Exception as e:
                print(f"  [resume] failed ({e}); restarting variant")
                accs, f1s = [], []

        for idx, subj in enumerate(unique_subjects):
            if idx < done:
                print(f"  Test Subject {idx+1}/{n_total} (id={subj}) [skipped]")
                continue
            print(f"  Test Subject {idx+1}/{n_total} (id={subj})")

            train_idx = [i for i, s in enumerate(ds.subject_ids) if s != subj]
            test_idx = [i for i, s in enumerate(ds.subject_ids) if s == subj]
            train_ds = torch.utils.data.Subset(ds, train_idx)
            test_ds = torch.utils.data.Subset(ds, test_idx)
            train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
            test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

            model = AMDGCN(
                n_channels=n_ch, n_regions=5, n_classes=n_cls,
                n_bands=5, hidden_dim=64, dropout=0.5,
                **variant_kwargs,
            ).to('cpu')
            trainer = Trainer(model, device='cpu')
            opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

            best_acc, best_f1, pc = 0.0, 0.0, 0
            for ep in range(epochs):
                trainer.train_epoch(train_ld, opt, crit)
                vm = trainer.evaluate(test_ld, crit)
                sched.step(vm['accuracy'])
                if vm['accuracy'] > best_acc:
                    best_acc = vm['accuracy']
                    best_f1 = vm['f1_score']
                    pc = 0
                else:
                    pc += 1
                if pc >= patience:
                    print(f"    Early stopping at epoch {ep+1}")
                    break

            accs.append(float(best_acc))
            f1s.append(float(best_f1))
            print(f"    Subject {subj}: Acc={best_acc:.4f}, F1={best_f1:.4f}")
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump({'accs': accs, 'f1s': f1s}, f, indent=2)

            del model, trainer, opt
            gc.collect()

        results[name] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'per_subject_accuracy': [float(a) for a in accs],
            'per_subject_f1': [float(f) for f in f1s],
        }
        print(f"  {name}: Acc={results[name]['mean_accuracy']:.4f} "
              f"+/- {results[name]['std_accuracy']:.4f}")
        with open(abl_ckpt, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"  [SAVED] Checkpoint updated ({len(results)}/{len(variants)} variants)")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Final save to a timestamped dir.
    ts = time.strftime('%Y%m%d_%H%M%S')
    abl_dir = os.path.join('outputs', f'SEEDIV_ablation_final_{ts}')
    os.makedirs(abl_dir, exist_ok=True)
    with open(os.path.join(abl_dir, 'ablation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {abl_dir}/ablation_results.json")

    del ds
    gc.collect()
    return results


def main():
    epochs, patience = 60, 20
    max_retries = 50

    for attempt in range(max_retries):
        try:
            results = run_ablation_optimized(DATA_DIR, epochs=epochs, patience=patience)
        except Exception as e:
            print(f"\n[WATCHDOG] ablation raised: {e!r}; retrying (attempt {attempt+1}/{max_retries})")
            traceback.print_exc()
            gc.collect()
            time.sleep(15)
            continue

        print("\n" + "=" * 70)
        print("SEED-IV ABLATION SUMMARY (60 ep / patience 20)")
        print("=" * 70)
        for name, r in results.items():
            print(f"  {name:28s}: {r['mean_accuracy']*100:5.2f}% +/- {r['std_accuracy']*100:4.2f}%  "
                  f"(F1 {r['mean_f1']*100:5.2f}% +/- {r['std_f1']*100:4.2f}%)")
        print("=" * 70)
        return

    print("\n[WATCHDOG] exceeded max retries -- aborting. Check logs.")


if __name__ == '__main__':
    main()
