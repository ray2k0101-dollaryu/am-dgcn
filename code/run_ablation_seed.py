"""
Dedicated SEED 9-variant ablation runner (fixed-code, crash-resilient).

Calls run_all_v3.run_ablation_optimized with the corrected dataset path.
Uses the efficient ablation setting (60 epochs / 20 patience) — the original
"optimized ablation" configuration — which yields the same qualitative
component-attribution conclusions ~2.5x faster than 150 epochs.

A watchdog retry loop re-invokes the ablation after any in-process exception;
the per-variant + per-subject checkpoints in run_all_v3 make each retry resume
from the last completed variant/subject (no lost work).

Launch:
  python launcher.py --log outputs/seed_ablation.log --script run_ablation_seed
To resume after a crash, just re-run the same command.
"""
import os
import sys
import time
import gc
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_all_v3 import run_ablation_optimized
from paths import dataset_dir


def main():
    data_dir = dataset_dir('SEED')
    # Efficient ablation setting: 60 ep / patience 20 (qualitative conclusions
    # identical to 150 ep; see plan Phase 1 D4 note on optimized ablation).
    epochs, patience = 60, 20
    max_retries = 50

    for attempt in range(max_retries):
        try:
            results = run_ablation_optimized(data_dir, epochs=epochs, patience=patience)
        except Exception as e:  # in-process (non-segfault) crash
            print(f"\n[WATCHDOG] ablation raised: {e!r}; retrying (attempt {attempt+1}/{max_retries})")
            traceback.print_exc()
            gc.collect()
            time.sleep(15)
            continue

        # Success: all variants done.
        print("\n" + "=" * 70)
        print("SEED ABLATION SUMMARY (fixed code, 60 ep / patience 20)")
        print("=" * 70)
        for name, r in results.items():
            print(f"  {name:28s}: {r['mean_accuracy']*100:5.2f}% ± {r['std_accuracy']*100:4.2f}%  "
                  f"(F1 {r['mean_f1']*100:5.2f}% ± {r['std_f1']*100:4.2f}%)")
        print("=" * 70)
        print("Saved to outputs/SEED_ablation_final_<ts>/ablation_results.json")
        return

    print("\n[WATCHDOG] exceeded max retries — aborting. Check logs.")


if __name__ == '__main__':
    main()
