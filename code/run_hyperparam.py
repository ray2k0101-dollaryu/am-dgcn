"""
Hyperparameter analysis experiments for AM-DGCN.
Runs SI (LOSO) on SEED with different hyperparameter values.

Usage:
    python run_hyperparam.py --param alpha --dataset SEED --epochs 100
    python run_hyperparam.py --param layers --dataset SEED --epochs 100
    python run_hyperparam.py --param hidden --dataset SEED --epochs 100
"""
import argparse
import json
import os
import gc
import sys
from datetime import datetime
from paths import dataset_dir

# Add parent dir to path and change working directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
os.chdir(_script_dir)


def run_hyperparam(param_name, dataset_name, data_dir, epochs, device, patience):
    """Run hyperparameter sweep for a specific parameter."""
    from run_experiment import load_dataset
    from am_dgcn_model import AMDGCN, Trainer

    # Load dataset once
    print(f"\nLoading {dataset_name}...")
    ds = load_dataset(dataset_name, data_dir)
    print(ds.summary())

    n_ch = ds.features[0].shape[0]
    n_classes = ds.n_classes

    # Define parameter configs
    if param_name == 'alpha':
        configs = [
            ('alpha_0.0', {'alpha_init': 0.0}),
            ('alpha_0.25', {'alpha_init': 0.25}),
            ('alpha_0.5', {'alpha_init': 0.5}),  # default
            ('alpha_0.75', {'alpha_init': 0.75}),
            ('alpha_1.0', {'alpha_init': 1.0}),
        ]
    elif param_name == 'layers':
        configs = [
            ('layers_1', {'n_gcn_layers': 1}),
            ('layers_2', {'n_gcn_layers': 2}),  # default
            ('layers_3', {'n_gcn_layers': 3}),
            ('layers_4', {'n_gcn_layers': 4}),
        ]
    elif param_name == 'hidden':
        configs = [
            ('hidden_32', {'hidden_dim': 32}),
            ('hidden_64', {'hidden_dim': 64}),  # default
            ('hidden_128', {'hidden_dim': 128}),
            ('hidden_256', {'hidden_dim': 256}),
        ]
    else:
        raise ValueError(f"Unknown param: {param_name}")

    # Output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('outputs', f'hyperparam_{param_name}_{dataset_name}_{ts}')
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'hyperparam_results.json')

    # Persistent checkpoint (survives crashes / relaunches, unlike the
    # timestamped output dir which is recreated on every launch).
    ckpt_path = os.path.join(
        'outputs', f'hyperparam_{param_name}_{dataset_name}_checkpoint.json')
    results = {}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"\n[RESUME] Checkpoint: {len(results)}/{len(configs)} configs done")
            for k, v in results.items():
                print(f"  [done] {k}: Acc={v['mean_accuracy']:.4f}")
        except Exception as e:
            print(f"[RESUME] Failed to read checkpoint ({e}), starting fresh")
            results = {}

    for name, kwargs in configs:
        if name in results:
            print(f"\n--- {name}: SKIP (already done) ---")
            continue

        print(f"\n{'=' * 70}")
        print(f"Config: {name} ({kwargs})")
        print(f"{'=' * 70}")

        # Base config — kwargs overrides defaults
        base_kwargs = dict(
            n_channels=n_ch,
            n_regions=5,
            n_classes=n_classes,
            n_bands=5,
            hidden_dim=64,
            dropout=0.5,
        )
        base_kwargs.update(kwargs)
        model = AMDGCN(**base_kwargs)
        trainer = Trainer(model, device=device)

        si_res = trainer.subject_independent_loso(
            ds,
            n_subjects=ds.n_subjects,
            epochs=epochs,
            lr=0.001,
            batch_size=64,
            patience=patience,
        )

        results[name] = {
            'config': kwargs,
            'mean_accuracy': float(si_res['mean_accuracy']),
            'std_accuracy': float(si_res['std_accuracy']),
            'mean_f1': float(si_res['mean_f1']),
            'std_f1': float(si_res['std_f1']),
        }
        print(f"  Acc: {si_res['mean_accuracy']:.4f} ± {si_res['std_accuracy']:.4f}")
        print(f"  F1:  {si_res['mean_f1']:.4f} ± {si_res['std_f1']:.4f}")

        # Count params
        n_params = sum(p.numel() for p in model.parameters())
        results[name]['n_params'] = n_params
        print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

        # Incremental save (both the run dir and the resume checkpoint)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Cleanup
        del model, trainer
        gc.collect()

    # Final write: ensure the run dir holds every config, including those
    # restored from the checkpoint and skipped this run.
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"HYPERPARAM SUMMARY: {param_name} ({dataset_name}, SI/LOSO)")
    print(f"{'=' * 70}")
    for name, _ in configs:
        r = results.get(name)
        if r:
            print(f"  {name:<14}: {r['mean_accuracy']*100:.2f}% "
                  f"+/- {r['std_accuracy']*100:.2f}%  "
                  f"(F1 {r['mean_f1']*100:.2f}%, {r['n_params']/1e6:.2f}M params)")
    print(f"{'=' * 70}")
    print(f"\nResults saved to {results_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="AM-DGCN hyperparameter analysis")
    p.add_argument('--param', required=True, choices=['alpha', 'layers', 'hidden'],
                   help="Which hyperparameter to sweep")
    p.add_argument('--dataset', default='SEED', choices=['SEED', 'SEED-IV', 'DEAP'])
    p.add_argument('--data_dir', default=None)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--device', default='cpu')
    p.add_argument('--patience', type=int, default=50)

    args = p.parse_args()

    data_dirs = {
        'DEAP': dataset_dir('DEAP'),
        'SEED': dataset_dir('SEED'),
        'SEED-IV': dataset_dir('SEED_IV'),
    }
    data_dir = args.data_dir or data_dirs.get(args.dataset, '.')

    run_hyperparam(args.param, args.dataset, data_dir,
                   args.epochs, args.device, args.patience)


if __name__ == '__main__':
    main()
