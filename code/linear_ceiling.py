"""Linear ceiling re-test after Phase 1 fixes.

Computes the accuracy attainable by a plain linear classifier (logistic
regression on standardized flattened DE features) under BOTH protocols used
for AM-DGCN, so the model can be compared against a fair linear reference:

  * SD (subject-dependent):   stratified 5-fold CV over all trials
  * SI (subject-independent): leave-one-subject-out

This is the central reference for the diagnostic framing of the paper: if the
GNN does not clearly exceed this ceiling, the graph machinery is not earning
its keep.

Usage (foreground PowerShell):
    python linear_ceiling.py --dataset SEED
    python linear_ceiling.py --dataset SEED-IV
    python linear_ceiling.py --dataset DEAP --label_type valence
"""
import argparse, json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from paths import dataset_dir

DATA_DIRS = {
    'SEED': dataset_dir('SEED'),
    'SEED-IV': dataset_dir('SEED_IV'),
    'DEAP': dataset_dir('DEAP'),
}


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, n_jobs=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='SEED',
                    choices=['SEED', 'SEED-IV', 'DEAP'])
    ap.add_argument('--label_type', default='valence',
                    choices=['valence', 'arousal'])
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--force', action='store_true',
                    help='recompute even if the output json already exists')
    args = ap.parse_args()

    tag = f"DEAP-{args.label_type}" if args.dataset == 'DEAP' else args.dataset
    os.makedirs('outputs', exist_ok=True)
    path = os.path.join('outputs', f'linear_ceiling_{tag}.json')
    if os.path.exists(path) and not args.force:
        print(f"[skip] {path} already exists; use --force to recompute")
        print("LINEAR_CEILING_DONE")
        return

    from run_experiment import load_dataset
    data_dir = args.data_dir or DATA_DIRS[args.dataset]

    t0 = time.time()
    if args.dataset == 'DEAP':
        ds = load_dataset('DEAP', data_dir, label_type=args.label_type)
    else:
        ds = load_dataset(args.dataset, data_dir)
    print(ds.summary())

    X = np.asarray([np.asarray(f, dtype=np.float32).reshape(-1)
                    for f in ds.features], dtype=np.float32)
    Y = np.asarray(ds.labels, dtype=np.int64).reshape(-1)
    S = np.asarray(ds.subject_ids).reshape(-1)
    print(f"[data] X={X.shape} Y={Y.shape} classes={np.bincount(Y)} "
          f"subjects={len(set(S.tolist()))}  ({time.time()-t0:.1f}s)")

    out = {'dataset': tag, 'n_trials': int(X.shape[0]),
           'n_features': int(X.shape[1]),
           'n_classes': int(len(np.unique(Y)))}

    # ---------- SD: stratified 5-fold ----------
    print("\n[SD] linear ceiling (stratified 5-fold)")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, f1s = [], []
    for k, (tr, te) in enumerate(skf.split(X, Y)):
        clf = make_clf().fit(X[tr], Y[tr])
        p = clf.predict(X[te])
        a = accuracy_score(Y[te], p); f = f1_score(Y[te], p, average='macro')
        accs.append(a); f1s.append(f)
        print(f"  fold {k+1}: acc={a:.4f} f1={f:.4f}")
    out['sd_accuracy_mean'] = float(np.mean(accs))
    out['sd_accuracy_std'] = float(np.std(accs))
    out['sd_f1_mean'] = float(np.mean(f1s))
    out['sd_f1_std'] = float(np.std(f1s))
    print(f"  => SD acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%  "
          f"f1 = {np.mean(f1s)*100:.2f}%")

    # ---------- SI: leave-one-subject-out ----------
    print("\n[SI] linear ceiling (leave-one-subject-out)")
    subs = sorted(set(S.tolist()))
    accs, f1s = [], []
    si_per_subject = []   # retained for paired significance tests (per held-out subject)
    si_subject_ids = []
    for i, s in enumerate(subs):
        tr = np.where(S != s)[0]; te = np.where(S == s)[0]
        if len(te) == 0 or len(np.unique(Y[tr])) < 2:
            continue
        clf = make_clf().fit(X[tr], Y[tr])
        p = clf.predict(X[te])
        a = accuracy_score(Y[te], p); f = f1_score(Y[te], p, average='macro')
        accs.append(a); f1s.append(f)
        si_per_subject.append(float(a))
        si_subject_ids.append(int(s) if isinstance(s, (int, float)) else str(s))
        print(f"  subj {s} ({i+1}/{len(subs)}): acc={a:.4f} f1={f:.4f}")
    out['si_accuracy_mean'] = float(np.mean(accs))
    out['si_accuracy_std'] = float(np.std(accs))
    out['si_f1_mean'] = float(np.mean(f1s))
    out['si_f1_std'] = float(np.std(f1s))
    out['si_per_subject'] = si_per_subject
    out['si_subject_ids'] = si_subject_ids
    print(f"  => SI acc = {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%  "
          f"f1 = {np.mean(f1s)*100:.2f}%")

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[saved] {path}")
    print(f"[total] {time.time()-t0:.1f}s")
    print("LINEAR_CEILING_DONE")


if __name__ == '__main__':
    main()
