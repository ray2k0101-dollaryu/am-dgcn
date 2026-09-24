"""
Detached runner for the three SEED hyperparameter sweeps (alpha -> layers -> hidden).

Designed to be launched by Windows Task Scheduler (schtasks), i.e. WITHOUT a console:
  - never touches sys.__stdout__ (which is None in a detached/pythonw context)
  - opens each log in APPEND mode so relaunches never truncate history
  - relies on run_hyperparam.py's checkpoint resume to skip finished configs
  - writes a heartbeat/status line so the guard automation can tell what is going on

Usage:
  python -X utf8 run_hp_all.py
  python -X utf8 run_hp_all.py --epochs 60 --patience 20
"""
import sys
import os
import io
import time
import traceback
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

SWEEPS = [
    # (param name, expected number of configs)
    ('alpha', 5),
    ('layers', 4),
    ('hidden', 4),
]

OUT = os.path.join(HERE, 'outputs')


def checkpoint_path(param, dataset):
    return os.path.join(OUT, f'hyperparam_{param}_{dataset}_checkpoint.json')


def done_count(param, dataset):
    """How many configs are already recorded in the checkpoint file."""
    import json
    p = checkpoint_path(param, dataset)
    if not os.path.exists(p):
        return 0
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return len(json.load(f))
    except Exception:
        return 0


class LogWriter:
    """Append-mode, line-flushed, console-free log sink."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8', errors='replace')

    def write(self, data):
        self.f.write(data)
        self.f.flush()

    def flush(self):
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def run_sweep(param, dataset, epochs, patience, master):
    """Run one sweep with stdout/stderr redirected to its own log file."""
    log_path = os.path.join(OUT, f'hp_{param}.log')
    sink = LogWriter(log_path)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sink
    sys.stderr = sink
    try:
        print(f'\n{"=" * 70}')
        print(f'[run_hp_all] START sweep={param} at {stamp()} (pid={os.getpid()})')
        print(f'{"=" * 70}\n')

        sys.argv = [
            'run_hyperparam.py',
            '--param', param,
            '--dataset', dataset,
            '--epochs', str(epochs),
            '--patience', str(patience),
        ]
        import run_hyperparam
        run_hyperparam.main()

        print(f'\n[run_hp_all] FINISHED sweep={param} at {stamp()}\n')
        return True
    except Exception:
        print(f'\n[run_hp_all] ERROR in sweep={param} at {stamp()}')
        traceback.print_exc()
        return False
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        sink.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='SEED')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=20)
    args = ap.parse_args()

    master = LogWriter(os.path.join(OUT, 'hp_runner.log'))

    def note(msg):
        master.write(f'[{stamp()}] {msg}\n')

    note(f'=== run_hp_all START pid={os.getpid()} '
         f'dataset={args.dataset} epochs={args.epochs} patience={args.patience} ===')

    for param, expected in SWEEPS:
        have = done_count(param, args.dataset)
        if have >= expected:
            note(f'sweep {param}: already complete ({have}/{expected}) -> SKIP')
            continue

        note(f'sweep {param}: {have}/{expected} done -> running')
        ok = run_sweep(param, args.dataset, args.epochs, args.patience, master)
        have = done_count(param, args.dataset)
        note(f'sweep {param}: returned ok={ok}, now {have}/{expected} done')

        if have < expected:
            note(f'sweep {param}: INCOMPLETE, stopping so a relaunch can resume here')
            master.close()
            sys.exit(1)

    note('=== ALL SWEEPS COMPLETE ===')
    master.close()


if __name__ == '__main__':
    main()
