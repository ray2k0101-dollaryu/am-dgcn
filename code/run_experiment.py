"""
Real Data Loaders for AM-DGCN: SEED, SEED-IV, and DEAP datasets.

Supports:
- SEED (3-class: positive/neutral/negative)
- SEED-IV (4-class: neutral/sad/fear/happy)  
- DEAP (binary: high/low valence, high/low arousal)

Usage:
    from run_experiment import load_dataset
    dataset = load_dataset('SEED', data_dir='D:/EEG_data/SEED')
    dataset = load_dataset('DEAP', data_dir='D:/EEG_data/DEAP')
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# Try importing scipy for .mat files
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed. SEED .mat loading will fail.")

# Try importing h5py for v7.3 .mat files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================================
# EEGEmotionDataset (unified PyTorch Dataset)
# ============================================================================

class EEGEmotionDataset(Dataset):
    """PyTorch Dataset for EEG emotion recognition."""

    def __init__(self, features: List[np.ndarray], labels: List[int],
                 subject_ids: List[int], channel_names: List[str],
                 plv_matrices: Optional[List[np.ndarray]] = None,
                 assignment_matrix: Optional[np.ndarray] = None):
        self.features = features
        self.labels = labels
        self.subject_ids = subject_ids
        self.channel_names = channel_names
        self.plv_matrices = plv_matrices
        self.assignment_matrix = assignment_matrix

        assert len(features) == len(labels) == len(subject_ids)
        if plv_matrices is not None:
            assert len(plv_matrices) == len(features)

        # Cache assignment matrix as tensor
        if self.assignment_matrix is not None:
            self._M = torch.tensor(self.assignment_matrix, dtype=torch.float32)
        else:
            n_ch = features[0].shape[0] if len(features) > 0 else 62
            self._M = torch.eye(n_ch, 5)

        self.n_classes = len(set(labels))
        self.n_subjects = len(set(subject_ids))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float32),
            'y': torch.tensor(self.labels[idx], dtype=torch.long),
            'plv': torch.tensor(self.plv_matrices[idx], dtype=torch.float32)
                   if self.plv_matrices is not None
                   else torch.eye(self.features[idx].shape[0]),
            'assignment_matrix': self._M,
        }

    def summary(self) -> str:
        lines = [
            f"Dataset: {len(self)} trials, {self.n_subjects} subjects, {self.n_classes} classes",
            f"Feature shape: {self.features[0].shape}",
        ]
        # Class distribution
        label_counts = defaultdict(int)
        for l in self.labels:
            label_counts[l] += 1
        for k in sorted(label_counts):
            lines.append(f"  Class {k}: {label_counts[k]} trials")
        return '\n'.join(lines)


def collate_fn(batch):
    """Custom collate to handle dict items."""
    return {
        'x': torch.stack([b['x'] for b in batch]),
        'y': torch.stack([b['y'] for b in batch]),
        'plv': torch.stack([b['plv'] for b in batch]),
        'assignment_matrix': batch[0]['assignment_matrix'],
    }


# ============================================================================
# SEED Data Loader
# ============================================================================

SEED_62CH = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4',
    'C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6',
    'TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
]

DEAP_32CH = [
    'Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5',
    'P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8',
    'CP6','CP2','C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz'
]


class SEEDLoader:
    """
    Loader for SEED and SEED-IV datasets (BCMI format).

    Expected directory structure:
        SEED/
        └── ExtractedFeatures/
            ├── 1_20131027.mat    # subject 1
            ├── 2_20140419.mat    # subject 2
            ├── ...
            └── 15_20150709.mat   # subject 15

    Each .mat file contains a struct with:
        - data: cell array {3 sessions}[15 trials], each trial is (62, n_samples)
        - label: cell array {3 sessions}[15 trials], each trial is scalar (+1/0/-1)
    """

    def __init__(self, data_dir: str, dataset_name: str = 'SEED',
                 preprocessor=None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name.upper()
        self.preprocessor = preprocessor
        self.channel_names = SEED_62CH
        self.fs = 200  # SEED sampling rate

    def _find_mat_files(self) -> List[Tuple[str, int]]:
        """Find all subject .mat files, return (filepath, session) tuples.
        session=0 for flat folders (SEED), 1/2/3 for session subfolders (SEED-IV).
        """
        results = []
        # Try common subdirectories
        search_dirs = [
            os.path.join(self.data_dir, 'eeg_feature_smooth'),
            os.path.join(self.data_dir, 'ExtractedFeatures'),
            os.path.join(self.data_dir, 'Preprocessed_EEG'),
            self.data_dir,
        ]
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            # Check for session subfolders (SEED-IV: 1/, 2/, 3/)
            sub_entries = sorted(os.listdir(d))
            session_dirs = [e for e in sub_entries
                            if os.path.isdir(os.path.join(d, e)) and e.isdigit()]
            if session_dirs:
                for sd in session_dirs:
                    session = int(sd)
                    sd_path = os.path.join(d, sd)
                    for fname in sorted(os.listdir(sd_path)):
                        if fname.endswith('.mat') and not fname.startswith('.'):
                            results.append((os.path.join(sd_path, fname), session))
                if results:
                    return results
            else:
                # Flat folder (SEED)
                for fname in sub_entries:
                    if fname.endswith('.mat') and not fname.startswith('.'):
                        results.append((os.path.join(d, fname), 0))
                if results:
                    return results
        return results

    def _parse_seed_struct(self, mat_data: dict) -> List[Dict]:
        """
        Parse SEED .mat struct into list of trial dicts.

        SEED .mat format (BCMI standard):
            The struct key varies by version. Common patterns:
            1. 'djc_eeg' or similar: struct with 'data' and 'label' fields
            2. Direct keys: 'data' (numpy object array) + 'label' (numpy object array)
            3. HDF5 format (v7.3): needs h5py to read

        Returns:
            List of dicts: [{'data': (62, N), 'label': 0, 'subject_id': s, 'session': ses, 'trial': t}, ...]
        """
        trials = []

        # Skip MATLAB metadata keys
        meta_keys = {'__header__', '__version__', '__globals__', '__function_workspace__'}

        # Find the actual data key
        data_key = None
        for key in mat_data:
            if key not in meta_keys and isinstance(mat_data[key], np.ndarray):
                # Check if this looks like the data container
                val = mat_data[key]
                if val.dtype == np.object_ or val.shape == (3,):
                    data_key = key
                    break
                # For HDF5-loaded data, look for references
                elif hasattr(val, 'shape') and len(val.shape) >= 1:
                    data_key = key
                    break

        if data_key is None:
            # Try keys that might contain the struct
            for key in mat_data:
                if key not in meta_keys:
                    data_key = key
                    break

        if data_key is None:
            raise ValueError(f"Cannot find data in .mat file. Keys: {list(mat_data.keys())}")

        container = mat_data[data_key]

        # Try to find label and data arrays
        labels_arr = None
        data_arr = None

        # Pattern 1: container is a struct array with 'data' and 'label' fields
        if isinstance(container, np.ndarray) and container.dtype.names is not None:
            field_names = list(container.dtype.names)
            for fn in field_names:
                lower = fn.lower()
                if 'label' in lower:
                    labels_arr = container[fn]
                elif 'data' in lower or 'eeg' in lower:
                    data_arr = container[fn]
            if data_arr is not None and labels_arr is not None:
                container = {'data': data_arr, 'label': labels_arr}

        # Pattern 2: container is a dict-like object
        if hasattr(container, '__getitem__') and not isinstance(container, np.ndarray):
            if 'label' in container:
                labels_arr = container['label']
            elif 'labels' in container:
                labels_arr = container['labels']
            if 'data' in container:
                data_arr = container['data']
            elif 'eeg' in container:
                data_arr = container['eeg']
        elif isinstance(container, np.ndarray):
            # Pattern 3: container is [3 sessions] object array
            # Each session is [15 trials] object array
            data_arr = container

        # If we still can't find it, try looking at top-level keys
        if data_arr is None:
            for key in mat_data:
                if key not in meta_keys and 'data' in key.lower():
                    data_arr = mat_data[key]
                    break
        if labels_arr is None:
            for key in mat_data:
                if key not in meta_keys and 'label' in key.lower():
                    labels_arr = mat_data[key]
                    break

        if data_arr is None:
            raise ValueError(f"Cannot locate data array in {list(mat_data.keys())}")

        # Parse sessions
        if isinstance(data_arr, np.ndarray) and data_arr.dtype == np.object_:
            # Standard format: 3 sessions
            n_sessions = data_arr.shape[0]
            if n_sessions > 10:  # Probably trials, not sessions
                n_sessions = 3
                trials_per_session = data_arr.shape[0] // 3
            else:
                trials_per_session = data_arr[0].shape[0] if hasattr(data_arr[0], 'shape') else 15

            for ses in range(min(n_sessions, 3)):
                session_data = data_arr[ses]
                session_labels = None
                if labels_arr is not None and labels_arr.dtype == np.object_:
                    if ses < labels_arr.shape[0]:
                        session_labels = labels_arr[ses]

                n_trials = (session_data.shape[0] if hasattr(session_data, 'shape')
                            else len(session_data) if hasattr(session_data, '__len__')
                            else 15)

                for t in range(min(n_trials, 30)):  # SEED: 15, SEED-IV: 24
                    trial_data = (session_data[t] if hasattr(session_data, '__getitem__')
                                  else session_data)
                    trial_data = np.array(trial_data, dtype=np.float64)

                    if trial_data.ndim == 1:
                        # Might need reshaping - try common shapes
                        n_ch = 62
                        n_samples = len(trial_data) // n_ch
                        if len(trial_data) % n_ch == 0:
                            trial_data = trial_data.reshape(n_ch, n_samples)
                        else:
                            continue

                    # Get label
                    label = 0
                    if session_labels is not None and hasattr(session_labels, '__getitem__'):
                        label = float(np.array(session_labels[t]).ravel()[0])
                    elif labels_arr is not None:
                        try:
                            label = float(np.array(labels_arr).ravel()[t + ses * n_trials])
                        except (IndexError, ValueError):
                            label = 0

                    # Convert SEED labels: +1→2 (positive), 0→1 (neutral), -1→0 (negative)
                    if self.dataset_name == 'SEED':
                        if label > 0.5:
                            label = 2
                        elif label < -0.5:
                            label = 0
                        else:
                            label = 1
                    else:
                        label = int(label)

                    trials.append({
                        'data': trial_data,
                        'label': label,
                        'session': ses,
                        'trial': t,
                    })

        return trials

    # SEED 15-试次固定标签(正1/中0/负-1)，3 分类；映射 -1->0, 0->1, 1->2
    SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    SEED_LABEL_MAP = {-1: 0, 0: 1, 1: 2}

    # SEED-IV 24-试次标签(0=neutral,1=sad,2=fear,3=happy)，按会话不同
    # 来源: SEED-IV/ReadMe.txt
    SEED_IV_SESSION_LABELS = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
    }

    def load_all(self) -> EEGEmotionDataset:
        """加载 SEED / SEED-IV 预提取 DE 特征(ExtractedFeatures)。
        每个 .mat = 1 被试 × 1 会话，含 de_LDS{1..N} 试次，形状 (n_ch, T, 5)。
        特征取时间均值 -> (n_ch, 5)；PLV 取通道间 DE 时间相关。
        """
        import re

        # Check cache first (SEED/SEED-IV .mat loading is fast but caching avoids re-parsing)
        cache_name = 'seed_cache.npz' if self.dataset_name == 'SEED' else 'seediv_cache.npz'
        cache_path = os.path.join('outputs', cache_name)
        if os.path.exists(cache_path):
            print(f"[CACHE] Loading {self.dataset_name} from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            # Coerce to per-trial float32 arrays (object-dtype caches cannot be
            # converted to torch tensors downstream).
            all_features = [np.asarray(f, dtype=np.float32) for f in data['features']]
            all_labels = list(data['labels'])
            all_subjects = list(data['subjects'])
            all_plvs = [np.asarray(p, dtype=np.float32) for p in data['plvs']]
            M = data['M'] if 'M' in data else None
            if isinstance(M, np.ndarray) and M.ndim == 0:
                M = M.item()
            print(f"  Loaded {len(all_features)} trials from cache.")
            return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                     self.channel_names, all_plvs, M)

        mat_files = self._find_mat_files()
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found in {self.data_dir}. "
                f"Expected SEED ExtractedFeatures/ or SEED-IV eeg_feature_smooth/ folder."
            )
        print(f"Found {len(mat_files)} subject files for {self.dataset_name}")

        all_features, all_labels, all_subjects, all_plvs = [], [], [], []
        M = None
        from am_dgcn_model import EEGPreprocessor
        try:
            M = EEGPreprocessor(fs=self.fs).build_assignment_matrix(self.channel_names).numpy()
        except Exception:
            M = None

        n_ch_expected = len(self.channel_names)
        for fpath, session in mat_files:
            base = os.path.basename(fpath)
            m = re.match(r"(\d+)_", base)
            subj = int(m.group(1)) if m else 0
            try:
                mat = sio.loadmat(fpath)
            except NotImplementedError:
                if HAS_H5PY:
                    mat = h5py.File(fpath, 'r')
                else:
                    print(f"    [SKIP] {base} v7.3 .mat requires h5py")
                    continue

            # 收集 de_LDS{1..N} 试次(主特征); 退而求其次用 de_movingAve
            trial_keys = []
            for k in mat.keys():
                if str(k).startswith("__"):
                    continue
                mm = re.match(r"de_LDS(\d+)$", str(k))
                if mm:
                    trial_keys.append((int(mm.group(1)), k))
            if not trial_keys:
                for k in mat.keys():
                    if str(k).startswith("__"):
                        continue
                    mm = re.match(r"de_movingAve(\d+)$", str(k))
                    if mm:
                        trial_keys.append((int(mm.group(1)), k))
            trial_keys.sort()
            print(f"  {base}: {len(trial_keys)} DE trials (subj={subj}, session={session})")

            for tidx, (tn, key) in enumerate(trial_keys):
                de = np.array(mat[key], dtype=np.float32)  # (n_ch, T, 5)
                if de.ndim != 3:
                    continue
                if de.shape[0] != n_ch_expected:
                    # 通道数不匹配(可能是 SEED-IV 62 通道); 尽量兼容
                    if de.shape[0] < 2:
                        continue
                feat = de.mean(axis=1)  # (n_ch, 5)
                # Functional connectivity proxy: SEED/SEED-IV provide only the
                # official pre-extracted DE features (no raw signal), so a true
                # phase-based PLV cannot be computed. We instead build a
                # BAND-AWARE connectivity matrix: compute cross-channel Pearson
                # correlation of the DE time series WITHIN each frequency band,
                # then average the absolute correlations across the 5 bands.
                n_ch_de, T_de, n_band_de = de.shape
                if T_de > 2:
                    band_mats = []
                    for b in range(n_band_de):
                        cb = np.corrcoef(de[:, :, b])  # (n_ch, n_ch) for band b
                        cb = np.nan_to_num(cb, nan=0.0)
                        band_mats.append(np.abs(cb))
                    c = np.mean(band_mats, axis=0).astype(np.float32)
                    np.fill_diagonal(c, 0.0)
                else:
                    c = np.eye(n_ch_de, dtype=np.float32)
                # 标签
                if self.dataset_name == 'SEED':
                    lab_raw = self.SEED_LABELS[tidx % len(self.SEED_LABELS)]
                    label = self.SEED_LABEL_MAP[lab_raw]
                else:
                    # SEED-IV: 4 分类，标签按会话不同
                    sess_labels = self.SEED_IV_SESSION_LABELS.get(session, [0])
                    label = sess_labels[tidx] if tidx < len(sess_labels) else 0
                all_features.append(feat)
                all_labels.append(label)
                all_subjects.append(subj)
                all_plvs.append(c)

        print(f"Total: {len(all_features)} valid trials loaded.")

        # Save cache for subsequent loads (dense float32 when shapes are uniform;
        # object-array fallback for ragged channel counts). Object arrays cannot
        # be converted to torch tensors, so dense storage is strongly preferred.
        if all_features:
            os.makedirs('outputs', exist_ok=True)
            M_save = M if M is not None else np.array(None, dtype=object)
            try:
                feat_arr = np.stack(all_features).astype(np.float32)
                plv_arr = np.stack(all_plvs).astype(np.float32)
                np.savez_compressed(cache_path,
                    features=feat_arr,
                    labels=np.array(all_labels),
                    subjects=np.array(all_subjects),
                    plvs=plv_arr,
                    M=M_save)
            except (ValueError, MemoryError):
                np.savez_compressed(cache_path,
                    features=np.array(all_features, dtype=object),
                    labels=np.array(all_labels),
                    subjects=np.array(all_subjects),
                    plvs=np.array(all_plvs, dtype=object),
                    M=M_save)
            print(f"[CACHE] Saved {self.dataset_name} cache ({len(all_features)} trials)")

        if not all_features:
            return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                     self.channel_names, None, M)
        return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                 self.channel_names, all_plvs, M)


# ============================================================================
# DEAP Data Loader
# ============================================================================

class DEAPLoader:
    """
    Loader for DEAP dataset.

    Expected directory structure:
        DEAP/
        └── data_preprocessed_python/
            ├── s01.dat
            ├── s02.dat
            ├── ...
            └── s32.dat

    Each .dat file is a Python pickle containing:
        data:   (40, 40, 8064)  = 40 trials × 40 channels × 8064 samples
        labels: (40, 4)         = 40 trials × [valence, arousal, dominance, liking]
    """

    def __init__(self, data_dir: str, label_type: str = 'valence',
                 preprocessor=None):
        """
        Args:
            data_dir: Path to DEAP dataset root (contains data_preprocessed_python/)
            label_type: 'valence' or 'arousal' for binary classification
        """
        self.data_dir = data_dir
        self.label_type = label_type
        self.preprocessor = preprocessor
        self.channel_names = DEAP_32CH
        self.fs = 128  # DEAP preprocessed sampling rate
        self.n_eeg_channels = 32  # First 32 are EEG, last 8 are peripheral

    def _find_dat_dir(self) -> str:
        """Find the data_preprocessed_python directory."""
        candidates = [
            os.path.join(self.data_dir, 'data_preprocessed_python'),
            self.data_dir,
        ]
        for d in candidates:
            if os.path.isdir(d):
                # Check if it contains s01.dat
                if os.path.exists(os.path.join(d, 's01.dat')):
                    return d
        raise FileNotFoundError(
            f"Cannot find data_preprocessed_python/ in {self.data_dir}. "
            f"Expected structure: DEAP/data_preprocessed_python/s01.dat ... s32.dat"
        )

    def load_all(self) -> EEGEmotionDataset:
        """Load and preprocess all 32 subjects."""
        # Check cache first (saves ~8 min on subsequent loads)
        cache_path = os.path.join('outputs', f'deap_cache_{self.label_type}.npz')
        # Also check old-style cache name (without label_type suffix)
        old_cache = os.path.join('outputs', 'deap_cache.npz')
        if os.path.exists(cache_path):
            print(f"[CACHE] Loading DEAP {self.label_type} from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            # Handle both dense (N, C, F) and object array formats
            feats = data['features']
            if feats.dtype == np.object_:
                # Old object array format
                all_features = [feats[i] for i in range(len(feats))]
            else:
                # Dense array format (N, C, F) — convert to list of per-trial arrays
                all_features = [feats[i] for i in range(len(feats))]
            plvs = data['plvs']
            if plvs.dtype == np.object_:
                all_plvs = [plvs[i] for i in range(len(plvs))]
            else:
                all_plvs = [plvs[i] for i in range(len(plvs))]
            all_labels = data['labels'].tolist()
            all_subjects = data['subjects'].tolist()
            M = data['M']
            if M.ndim == 0:  # Was saved as np.array(None)
                M = None
            print(f"  Loaded {len(all_features)} trials from cache.")
            return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                     self.channel_names, all_plvs, M)
        elif os.path.exists(old_cache):
            # Old-format cache: keys are X, Y, PLV — check if compatible
            print(f"[CACHE] Found old-style cache: {old_cache}")
            data = np.load(old_cache, allow_pickle=True)
            X = data['X']
            # Check if features have correct shape (n_bands=5)
            if X.ndim == 3 and X.shape[-1] == 5:
                print(f"  Old cache compatible (X shape: {X.shape}), loading...")
                Y = data['Y']
                PLV = data['PLV']
                n_trials = len(X)
                all_features = [X[i] for i in range(n_trials)]
                all_labels = Y.tolist()
                all_subjects = []
                for s in range(32):
                    all_subjects.extend([s + 1] * 40)
                all_subjects = all_subjects[:n_trials]
                all_plvs = [PLV[i] for i in range(n_trials)]
                from am_dgcn_model import EEGPreprocessor
                M = EEGPreprocessor(fs=self.fs).build_assignment_matrix(self.channel_names).numpy()
                print(f"  Loaded {n_trials} trials from old cache.")
                np.savez_compressed(cache_path,
                    features=np.array(all_features, dtype=object),
                    labels=np.array(all_labels),
                    subjects=np.array(all_subjects),
                    plvs=np.array(all_plvs, dtype=object),
                    M=M)
                print(f"[CACHE] Saved DEAP {self.label_type} cache in new format.")
                return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                         self.channel_names, all_plvs, M)
            else:
                print(f"  Old cache incompatible (X shape: {X.shape}, expected (*, *, 5)). Ignoring.")

        from am_dgcn_model import EEGPreprocessor
        if self.preprocessor is None:
            self.preprocessor = EEGPreprocessor(fs=self.fs)

        dat_dir = self._find_dat_dir()
        dat_files = sorted([f for f in os.listdir(dat_dir)
                           if f.startswith('s') and f.endswith('.dat')])

        if not dat_files:
            raise FileNotFoundError(f"No sXX.dat files found in {dat_dir}")

        print(f"Found {len(dat_files)} subject files for DEAP ({self.label_type})")

        all_features = []
        all_labels = []
        all_subjects = []
        all_plvs = []

        M = self.preprocessor.build_assignment_matrix(self.channel_names).numpy()

        for subj_idx, fname in enumerate(dat_files):
            fpath = os.path.join(dat_dir, fname)
            print(f"  Processing {fname} ({subj_idx + 1}/{len(dat_files)})")

            with open(fpath, 'rb') as f:
                raw = pickle.load(f, encoding='latin1')

            data = raw['data']    # (40, 40, 8064)
            labels = raw['labels']  # (40, 4)

            # Extract EEG channels only (first 32)
            eeg_data = data[:, :self.n_eeg_channels, :]  # (40, 32, 8064)

            # Baseline correction: subtract mean of first 3 seconds (384 samples at 128Hz)
            baseline = eeg_data[:, :, :384].mean(axis=2, keepdims=True)
            eeg_data = eeg_data - baseline

            # Label: valence column (index 0) or arousal column (index 1)
            label_col = 0 if self.label_type == 'valence' else 1
            trial_labels = labels[:, label_col]  # (40,)

            # Binary classification: > 5 → 1 (high), <= 5 → 0 (low)
            binary_labels = (trial_labels > 5).astype(int)

            for trial_idx in range(40):
                trial_data = eeg_data[trial_idx]  # (32, 8064)
                label = binary_labels[trial_idx]

                try:
                    processed = self.preprocessor.process_trial(
                        trial_data, self.channel_names, compute_plv_flag=True
                    )
                    all_features.append(processed['de_features'].numpy())
                    all_labels.append(int(label))
                    all_subjects.append(subj_idx)
                    all_plvs.append(processed['plv'].numpy())
                except Exception as e:
                    print(f"    [WARN] Trial {trial_idx} failed: {e}")
                    continue

        print(f"Total: {len(all_features)} valid trials loaded.")
        # Save cache for subsequent loads (use float32 dense arrays to save memory)
        os.makedirs('outputs', exist_ok=True)
        try:
            # Try dense stack (works if all features have same shape)
            feat_arr = np.stack(all_features).astype(np.float32)
            plv_arr = np.stack(all_plvs).astype(np.float32)
            np.savez_compressed(cache_path,
                                features=feat_arr,
                                labels=np.array(all_labels),
                                subjects=np.array(all_subjects),
                                plvs=plv_arr,
                                M=M.astype(np.float32) if M is not None else M)
        except (ValueError, MemoryError):
            # Fallback: object arrays (uses more memory but handles ragged data)
            np.savez_compressed(cache_path,
                                features=np.array(all_features, dtype=object),
                                labels=np.array(all_labels),
                                subjects=np.array(all_subjects),
                                plvs=np.array(all_plvs, dtype=object),
                                M=M)
        print(f"[CACHE] Saved DEAP {self.label_type} cache ({len(all_features)} trials)")
        return EEGEmotionDataset(all_features, all_labels, all_subjects,
                                 self.channel_names, all_plvs, M)


# ============================================================================
# Unified Loader
# ============================================================================

def load_dataset(dataset_name: str, data_dir: str, **kwargs) -> EEGEmotionDataset:
    """
    Unified dataset loader.

    Args:
        dataset_name: 'SEED', 'SEED-IV', or 'DEAP'
        data_dir: Path to dataset root directory
        **kwargs: Additional args (e.g., label_type='valence' for DEAP)

    Returns:
        EEGEmotionDataset ready for training
    """
    name = dataset_name.upper()

    if name in ('SEED', 'SEED-IV'):
        loader = SEEDLoader(data_dir, dataset_name=name)
        return loader.load_all()
    elif name == 'DEAP':
        label_type = kwargs.get('label_type', 'valence')
        loader = DEAPLoader(data_dir, label_type=label_type)
        return loader.load_all()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'SEED', 'SEED-IV', or 'DEAP'.")


# ============================================================================
# Experiment Runner
# ============================================================================

def run_full_experiment(config: Dict):
    """
    Run complete experiment: SD + SI on specified dataset.

    Args:
        config: {
            'dataset_name': 'SEED' | 'SEED-IV' | 'DEAP',
            'data_dir': str,
            'output_dir': str,
            'n_channels': int,
            'n_regions': int,
            'n_classes': int,
            'n_bands': int,
            'hidden_dim': int,
            'dropout': float,
            'epochs': int,
            'lr': float,
            'batch_size': int,
            'n_folds': int,
        }
    """
    from am_dgcn_model import AMDGCN, Trainer

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.get('output_dir', f'experiments/{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("AM-DGCN Experiment Runner")
    print("=" * 70)
    print(f"Dataset: {config['dataset_name']}")
    print(f"Data dir: {config['data_dir']}")
    print(f"Output: {output_dir}")

    # ---- Load Data ----
    print("\n[1/4] Loading dataset...")
    dataset = load_dataset(
        config['dataset_name'],
        config['data_dir'],
        label_type=config.get('label_type', 'valence')
    )
    print(dataset.summary())

    # Auto-detect n_classes if not specified
    n_classes = config.get('n_classes', dataset.n_classes)
    n_channels = dataset.features[0].shape[0]

    # ---- Train SD ----
    print("\n[2/4] Subject-Dependent 5-Fold CV...")
    model = AMDGCN(
        n_channels=n_channels,
        n_regions=config.get('n_regions', 5),
        n_classes=n_classes,
        n_bands=config.get('n_bands', 5),
        hidden_dim=config.get('hidden_dim', 64),
        dropout=config.get('dropout', 0.5),
    )
    trainer = Trainer(model, device=config.get('device', 'cpu'))

    sd_results = trainer.subject_dependent_cv(
        dataset,
        n_folds=config.get('n_folds', 5),
        epochs=config.get('epochs', 100),
        lr=config.get('lr', 0.001),
        batch_size=config.get('batch_size', 64),
        patience=config.get('patience', 40),
    )

    print(f"\n  SD Results:")
    print(f"    Accuracy: {sd_results['mean_accuracy']:.4f} ± {sd_results['std_accuracy']:.4f}")
    print(f"    F1-Score:  {sd_results['mean_f1']:.4f} ± {sd_results['std_f1']:.4f}")

    # ---- Train SI ----
    print("\n[3/4] Subject-Independent LOSO...")
    model_si = AMDGCN(
        n_channels=n_channels,
        n_regions=config.get('n_regions', 5),
        n_classes=n_classes,
        n_bands=config.get('n_bands', 5),
        hidden_dim=config.get('hidden_dim', 64),
        dropout=config.get('dropout', 0.5),
    )
    trainer_si = Trainer(model_si, device=config.get('device', 'cpu'))

    si_results = trainer_si.subject_independent_loso(
        dataset,
        n_subjects=dataset.n_subjects,
        epochs=config.get('epochs', 100),
        lr=config.get('lr', 0.001),
        batch_size=config.get('batch_size', 64),
        patience=config.get('patience', 40),
    )

    print(f"\n  SI Results:")
    print(f"    Accuracy: {si_results['mean_accuracy']:.4f} ± {si_results['std_accuracy']:.4f}")
    print(f"    F1-Score:  {si_results['mean_f1']:.4f} ± {si_results['std_f1']:.4f}")

    # ---- Save Results ----
    print("\n[4/4] Saving results...")
    all_results = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, list, dict)) else v
                   for k, v in config.items()},
        'dataset_summary': {
            'n_trials': len(dataset),
            'n_subjects': dataset.n_subjects,
            'n_classes': n_classes,
            'class_distribution': {int(k): v for k, v in
                                   zip(*np.unique(dataset.labels, return_counts=True))},
        },
        'subject_dependent': {
            'mean_accuracy': float(sd_results['mean_accuracy']),
            'std_accuracy': float(sd_results['std_accuracy']),
            'mean_f1': float(sd_results['mean_f1']),
            'std_f1': float(sd_results['std_f1']),
            'per_fold_accuracy': [float(a) for a in sd_results['accuracy']],
            'per_fold_f1': [float(f) for f in sd_results['f1_score']],
        },
        'subject_independent': {
            'mean_accuracy': float(si_results['mean_accuracy']),
            'std_accuracy': float(si_results['std_accuracy']),
            'mean_f1': float(si_results['mean_f1']),
            'std_f1': float(si_results['std_f1']),
            'per_subject_accuracy': [float(a) for a in si_results['accuracy']],
            'per_subject_f1': [float(f) for f in si_results['f1_score']],
        },
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(output_dir, 'model_sd.pt'))
    torch.save(model_si.state_dict(), os.path.join(output_dir, 'model_si.pt'))

    print(f"\nResults saved to {output_dir}/")
    print("=" * 70)
    return all_results


def run_ablation(config: Dict, dataset: EEGEmotionDataset):
    """
    Run ablation study comparing model variants.

    Variants tested:
    1. Full AM-DGCN (all components enabled)
    2. w/o Multi-Scale (only electrode-level)
    3. w/o Adaptive Adjacency (fixed distance-based)
    4. w/o PLV (only learnable adjacency)
    5. w/o Cross-Scale Message Passing
    6. w/o Scale Attention (equal weights)
    7. w/o Linear Shortcut (no linear feature path)
    8. Single Scale (GCN) (vanilla GCN, all enhancements off)
    """
    from am_dgcn_model import AMDGCN, Trainer

    n_channels = dataset.features[0].shape[0]
    n_classes = dataset.n_classes

    variants = {
        'Full AM-DGCN': {},
        'w/o Multi-Scale': {'use_multi_scale': False},
        'w/o Adaptive Adjacency': {'use_adaptive_adj': False},
        'w/o PLV': {'use_plv': False},
        'w/o Cross-Scale MP': {'use_cross_scale': False},
        'w/o Scale Attention': {'use_scale_attention': False},
        'w/o Linear Shortcut': {'use_linear_shortcut': False},
        # All graph-level components off but linear shortcut kept ON — isolates
        # the linear path's marginal value vs the pure-GCN baseline below.
        'Single Scale (GCN+Linear)': {
            'use_multi_scale': False,
            'use_adaptive_adj': False,
            'use_plv': False,
            'use_cross_scale': False,
            'use_scale_attention': False,
            'use_linear_shortcut': True,
        },
        'Single Scale (GCN)': {
            'use_multi_scale': False,
            'use_adaptive_adj': False,
            'use_plv': False,
            'use_cross_scale': False,
            'use_scale_attention': False,
            'use_linear_shortcut': False,
        },
    }

    results = {}
    print("\n" + "=" * 70)
    print("Ablation Study")
    print("=" * 70)

    for name, variant_kwargs in variants.items():
        print(f"\n--- {name} ---")
        model = AMDGCN(
            n_channels=n_channels,
            n_regions=config.get('n_regions', 5),
            n_classes=n_classes,
            n_bands=config.get('n_bands', 5),
            hidden_dim=config.get('hidden_dim', 64),
            dropout=config.get('dropout', 0.5),
            **variant_kwargs,
        )
        trainer = Trainer(model, device=config.get('device', 'cpu'))

        # Subject-independent evaluation (more discriminative for ablation)
        si_results = trainer.subject_independent_loso(
            dataset,
            n_subjects=dataset.n_subjects,
            epochs=config.get('epochs', 100),
            lr=config.get('lr', 0.001),
            batch_size=config.get('batch_size', 64),
            patience=config.get('patience', 40),
        )
        results[name] = {
            'mean_accuracy': float(si_results['mean_accuracy']),
            'std_accuracy': float(si_results['std_accuracy']),
            'mean_f1': float(si_results['mean_f1']),
            'std_f1': float(si_results['std_f1']),
        }
        print(f"  Acc: {si_results['mean_accuracy']:.4f} ± {si_results['std_accuracy']:.4f}")
        print(f"  F1:  {si_results['mean_f1']:.4f} ± {si_results['std_f1']:.4f}")

        # Free memory between variants
        del model, trainer
        import gc; gc.collect()

        # Incremental save (in case of crash)
        output_dir = config.get('output_dir', 'experiments/ablation')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # Final save
    output_dir = config.get('output_dir', 'experiments/ablation')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation results saved to {output_dir}/ablation_results.json")
    return results


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == '__main__':
    print("Data Loader Test")
    print("=" * 50)

    # Test with synthetic data
    from am_dgcn_model import EEGPreprocessor

    preprocessor = EEGPreprocessor(fs=200)

    # Create synthetic SEED-like data
    print("\nCreating synthetic data for testing...")
    loader = SEEDLoader(data_dir='.', dataset_name='SEED', preprocessor=preprocessor)

    # Since no real .mat files, test with manual trial
    test_data = np.random.randn(62, 4000)
    processed = preprocessor.process_trial(test_data, SEED_62CH, compute_plv_flag=True)
    print(f"  DE features shape: {processed['de_features'].shape}")
    print(f"  PLV shape: {processed['plv'].shape}")
    print(f"  Assignment matrix shape: {processed['assignment_matrix'].shape}")
    print("OK - Preprocessor works!")

    # Test DEAP pickle loading concept
    print("\nDEAP loader ready:")
    print("  data_preprocessed_python/s01.dat ~ s32.dat")
    print("  Format: pickle with {'data': (40,40,8064), 'labels': (40,4)}")
    print("  EEG channels: first 32 of 40")
    print("  Baseline: subtract first 3 seconds (384 samples @ 128Hz)")
    print("  Binary labels: valence/arousal > 5 → 1, else 0")

    print("\n" + "=" * 50)
    print("Ready! Usage:")
    print("  dataset = load_dataset('SEED', data_dir='D:/EEG_data/SEED')")
    print("  dataset = load_dataset('DEAP', data_dir='D:/EEG_data/DEAP', label_type='valence')")
    print("  results = run_full_experiment(config)")
