"""
Run all remaining experiments sequentially with resume capability.
1. DEAP arousal SI (with checkpoint per subject) [DONE]
2. SEED ablation (9 variants, 150 epochs, patience=50; Full AM-DGCN uses existing SI result)
3. SEED hyperparams (alpha/layers/hidden, 150 epochs, patience=50)

Usage:
    python launcher.py --log outputs/run_all_v3.log --script run_all_v3
"""
import os
import sys
import json
import time
import gc
from paths import dataset_dir

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
os.chdir(_script_dir)

ABL_CKPT = os.path.join('outputs', 'seed_ablation_checkpoint.json')


def run_tag():
    """Optional suffix so a multi-seed repeat gets its own files.

    Empty by default, in which case every path below is byte-identical to the
    single-seed behaviour the 2026-09-18 ablation used. A non-empty tag (e.g.
    '_seed20260919') gives that round its own checkpoint and scratch directory,
    so a repeat can never overwrite or accidentally resume into the
    authoritative run.
    """
    return os.environ.get('AMDGCN_RUN_TAG', '')


def tagged(base, tag=None):
    tag = run_tag() if tag is None else tag
    root, ext = os.path.splitext(base)
    return root + tag + ext


def run_deap_arousal_si():
    """Phase 0: DEAP arousal SI with checkpoint/resume."""
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from run_experiment import load_dataset, collate_fn
    from am_dgcn_model import AMDGCN, Trainer

    CKPT = os.path.join('outputs', 'deap_arousal_si_checkpoint.json')
    RESULT = os.path.join('outputs', 'deap_arousal_si_result.json')

    if os.path.exists(RESULT):
        with open(RESULT, 'r') as f:
            r = json.load(f)
        print(f"[SKIP] DEAP arousal SI already done: Acc={r['mean_accuracy']:.4f}")
        return r

    checkpoint = None
    if os.path.exists(CKPT):
        with open(CKPT, 'r') as f:
            checkpoint = json.load(f)
        print(f"[RESUME] {checkpoint['n_completed']}/{checkpoint['n_total']} done")

    ds = load_dataset('DEAP', dataset_dir('DEAP'), label_type='arousal')
    print(ds.summary())
    n_ch = ds.features[0].shape[0]
    n_cls = ds.n_classes
    unique_subjects = sorted(set(ds.subject_ids))
    n_total = len(unique_subjects)

    if checkpoint and checkpoint['n_completed'] > 0:
        accs = checkpoint['accuracies']
        f1s = checkpoint['f1_scores']
        start = checkpoint['n_completed']
    else:
        accs, f1s, start = [], [], 0

    for idx in range(start, n_total):
        subj = unique_subjects[idx]
        print(f"Test Subject {idx+1}/{n_total} (id={subj})")

        train_idx = [i for i, s in enumerate(ds.subject_ids) if s != subj]
        test_idx = [i for i, s in enumerate(ds.subject_ids) if s == subj]
        train_ds = torch.utils.data.Subset(ds, train_idx)
        test_ds = torch.utils.data.Subset(ds, test_idx)
        train_ld = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
        test_ld = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = AMDGCN(n_channels=n_ch, n_regions=5, n_classes=n_cls,
                       n_bands=5, hidden_dim=64, dropout=0.5).to('cpu')
        trainer = Trainer(model, device='cpu')
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)

        best_acc, best_f1, pc = 0.0, 0.0, 0
        for ep in range(100):
            trainer.train_epoch(train_ld, opt, crit)
            vm = trainer.evaluate(test_ld, crit)
            sched.step(vm['accuracy'])
            if vm['accuracy'] > best_acc:
                best_acc = vm['accuracy']
                best_f1 = vm['f1_score']
                pc = 0
            else:
                pc += 1
            if pc >= 40:
                print(f"  Early stopping at epoch {ep+1}")
                break

        accs.append(float(best_acc))
        f1s.append(float(best_f1))
        print(f"  Subject {subj}: Acc={best_acc:.4f}, F1={best_f1:.4f}")

        with open(CKPT, 'w') as f:
            json.dump({'accuracies': accs, 'f1_scores': f1s,
                       'n_completed': len(accs), 'n_total': n_total,
                       'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        del model, trainer, opt
        gc.collect()

    result = {
        'mean_accuracy': float(np.mean(accs)), 'std_accuracy': float(np.std(accs)),
        'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s)),
        'per_subject_accuracy': accs, 'per_subject_f1': f1s,
    }
    with open(RESULT, 'w') as f:
        json.dump(result, f, indent=2)
    if os.path.exists(CKPT):
        os.remove(CKPT)
    print(f"DEAP arousal SI done: Acc={result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}")
    del ds
    gc.collect()
    return result


def run_ablation_optimized(data_dir, epochs=150, patience=50):
    """Phase 1: SEED ablation with Full AM-DGCN pre-filled and reduced patience."""
    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import DataLoader
    from run_experiment import load_dataset, collate_fn
    from am_dgcn_model import AMDGCN, Trainer

    abl_ckpt = tagged(ABL_CKPT)
    tmp_dir = os.path.join('outputs', 'seed_ablation_tmp' + run_tag())
    print(f"[paths] checkpoint = {abl_ckpt}")
    print(f"[paths] scratch    = {tmp_dir}")
    print(f"[paths] run tag    = {run_tag()!r}")

    ds = load_dataset('SEED', data_dir)
    print(ds.summary())
    n_ch = ds.features[0].shape[0]
    n_cls = ds.n_classes
    unique_subjects = sorted(set(ds.subject_ids))
    n_total = len(unique_subjects)

    # All 8 variants
    variants = {
        'Full AM-DGCN': {},
        'w/o Multi-Scale': {'use_multi_scale': False},
        'w/o Adaptive Adjacency': {'use_adaptive_adj': False},
        'w/o PLV': {'use_plv': False},
        'w/o Cross-Scale MP': {'use_cross_scale': False},
        'w/o Scale Attention': {'use_scale_attention': False},
        'w/o Linear Shortcut': {'use_linear_shortcut': False},
        # Minimal GNN + linear shortcut (all graph-level components off but the
        # linear path kept ON). Comparing this against 'Single Scale (GCN)'
        # isolates the marginal value of the linear shortcut; comparing against
        # 'Full AM-DGCN' isolates the total value of all GNN components.
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
        # Pure linear shortcut, NO GCN pathway at all. Isolates whether the
        # electrode GCN adds any accuracy beyond the shortcut+MLP head. If this
        # matches 'Single Scale (GCN+Linear)', the graph pathway is redundant.
        'Linear Only (no GCN)': {
            'use_gcn_pathway': False, 'use_linear_shortcut': True,
        },
    }

    # Load checkpoint (per-variant results)
    results = {}
    if os.path.exists(abl_ckpt):
        with open(abl_ckpt, 'r') as f:
            results = json.load(f)
        print(f"[RESUME] Ablation checkpoint: {len(results)}/{len(variants)} variants done")
        for name in results:
            print(f"  ✓ {name}: Acc={results[name]['mean_accuracy']:.4f}")

    # NOTE: Full AM-DGCN is intentionally NOT pre-filled here. All 9 variants
    # must use identical training settings (60 ep / 20 patience for the efficient
    # ablation) so the attribution table is internally consistent. The 150-ep
    # SEED SI headline number lives separately in outputs/seed_fixed_sdsi/.
    if False and 'Full AM-DGCN' not in results:
        seed_si_path = os.path.join('outputs', 'seed_fixed_sdsi', 'si_results.json')
        if os.path.exists(seed_si_path):
            with open(seed_si_path, 'r') as f:
                si = json.load(f)
            results['Full AM-DGCN'] = {
                'mean_accuracy': si['mean_accuracy'],
                'std_accuracy': si['std_accuracy'],
                'mean_f1': si['mean_f1'],
                'std_f1': si['std_f1'],
            }
            print(f"[PREFILL] Full AM-DGCN = {si['mean_accuracy']:.4f} ± {si['std_accuracy']:.4f} (from existing SEED SI)")
            with open(abl_ckpt, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(f"[WARNING] SEED SI results not found at {seed_si_path}, will run Full AM-DGCN from scratch")

    # Per-subject scratch dir so a crash mid-variant only loses in-progress subjects.
    os.makedirs(tmp_dir, exist_ok=True)
    def _safe(n):
        return (n.replace(' ', '_').replace('(', '').replace(')', '')
                 .replace('/', '_').replace('\\', '_'))

    # Run remaining variants
    for name, variant_kwargs in variants.items():
        if name in results:
            print(f"\n--- {name}: SKIP (already done) ---")
            continue

        print(f"\n--- {name} ---")
        tmp_path = os.path.join(tmp_dir, _safe(name) + '.json')
        accs, f1s = [], []
        # Resume partial subjects for this variant.
        done = 0
        if os.path.exists(tmp_path):
            try:
                with open(tmp_path, 'r') as f:
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
            # Save per-subject progress immediately.
            with open(tmp_path, 'w') as f:
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
        print(f"  {name}: Acc={results[name]['mean_accuracy']:.4f} ± {results[name]['std_accuracy']:.4f}")

        # Save checkpoint after each variant
        with open(abl_ckpt, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  [SAVED] Checkpoint updated ({len(results)}/{len(variants)} variants)")
        # Clean up per-subject scratch for this variant.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Final save
    ts = time.strftime('%Y%m%d_%H%M%S')
    abl_dir = os.path.join('outputs', f'SEED_ablation_final_{ts}')
    os.makedirs(abl_dir, exist_ok=True)
    with open(os.path.join(abl_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {abl_dir}/ablation_results.json")

    del ds
    gc.collect()
    return results


def main():
    from run_hyperparam import run_hyperparam

    all_results = {}

    # Phase 0: DEAP arousal SI (with resume)
    print("\n" + "=" * 70)
    print("PHASE 0: DEAP arousal SI")
    print("=" * 70)
    all_results['deap_arousal_si'] = run_deap_arousal_si()

    # Phase 1: SEED Ablation (unified: epochs=150, patience=50)
    print("\n" + "=" * 70)
    print("PHASE 1: SEED Ablation (9 variants, 150 epochs, patience=50)")
    print("=" * 70)
    data_dir = dataset_dir('SEED')
    all_results['ablation'] = run_ablation_optimized(data_dir, epochs=150, patience=50)

    # Phase 2: Alpha sweep (unified: epochs=150, patience=50)
    print("\n" + "=" * 70)
    print("PHASE 2: Alpha sweep (5 values, 150 epochs, patience=50)")
    print("=" * 70)
    all_results['hyperparam_alpha'] = run_hyperparam('alpha', 'SEED', data_dir, 150, 'cpu', 50)
    gc.collect()

    # Phase 3: GCN layers (unified: epochs=150, patience=50)
    print("\n" + "=" * 70)
    print("PHASE 3: GCN layers sweep (4 values, 150 epochs, patience=50)")
    print("=" * 70)
    all_results['hyperparam_layers'] = run_hyperparam('layers', 'SEED', data_dir, 150, 'cpu', 50)
    gc.collect()

    # Phase 4: Hidden dim (unified: epochs=150, patience=50)
    print("\n" + "=" * 70)
    print("PHASE 4: Hidden dim sweep (4 values, 150 epochs, patience=50)")
    print("=" * 70)
    all_results['hyperparam_hidden'] = run_hyperparam('hidden', 'SEED', data_dir, 150, 'cpu', 50)
    gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    with open(os.path.join('outputs', 'all_remaining_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Summary saved to outputs/all_remaining_results.json")


if __name__ == '__main__':
    main()
