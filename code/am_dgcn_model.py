"""
AM-DGCN: Adaptive Multi-Scale Dynamic Graph Convolutional Network
for EEG-Based Emotion Recognition

Author: Lei Yu et al.
Framework: PyTorch + PyTorch Geometric
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import DataLoader  # dataset returns dicts; use standard collate
from scipy.signal import stft, butter, filtfilt, hilbert
from scipy.stats import entropy
from typing import Tuple, List, Optional, Dict
import json
import warnings
warnings.filterwarnings('ignore')

# mne is optional - only needed for advanced preprocessing
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# ============================================================================
# Standard 10-20 electrode geometry (for physically-meaningful adjacency)
# ============================================================================

# Canonical channel orders (must match the loaders in run_experiment.py)
SEED_62CH = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

DEAP_32CH = [
    'FP1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
    'P7', 'P3', 'PZ', 'PO3', 'O1', 'OZ', 'O2', 'PO4', 'P4', 'P8',
    'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'FZ', 'CZ'
]

# Anterior->posterior "row" index per electrode prefix (positive = anterior).
_AP_ROW = {
    'FP': 4.0, 'AF': 3.0, 'F': 2.0, 'FT': 1.0, 'FC': 1.0,
    'T': 0.0, 'C': 0.0, 'TP': -1.0, 'CP': -1.0, 'P': -2.0,
    'PO': -3.0, 'O': -4.0, 'I': -4.5, 'CB': -4.5,
}
# Left->right "column" index per trailing electrode index (positive = right).
_LR_COL = {'7': -4.0, '5': -3.0, '3': -2.0, '1': -1.0,
           'Z': 0.0, '2': 1.0, '4': 2.0, '6': 3.0, '8': 4.0}
# Degrees of scalp geodesic (polar angle from vertex Cz) per grid unit.
_ECC_STEP_DEG = 22.5


def _electrode_3d_position(name: str) -> Optional[Tuple[float, float, float]]:
    """
    Map a 10-20/10-10 electrode label to a point on the unit sphere via an
    azimuthal-equidistant projection of the standard scalp layout.

    Each electrode has a grid position (row = anterior-posterior,
    col = left-right) in the international 10-20 system. We treat the grid
    as a flat topographic map centered at the vertex Cz, convert it to a
    polar angle (eccentricity from Cz) and azimuth, then place it on the
    unit sphere. Cz is the north pole (0, 0, 1); +x is right, +y is
    anterior, +z is superior. Unlike a two-rotation model this does NOT
    collapse lateral separation near the anterior/posterior extremes, and
    unlike an electrode-index proxy it yields physically meaningful
    inter-electrode distances.
    """
    n = name.upper().strip()
    # Split alphabetic prefix from the trailing index (digit or Z).
    prefix, idx = '', ''
    for ch in n:
        if ch.isdigit() or ch == 'Z':
            idx += ch
        else:
            prefix += ch
    if prefix not in _AP_ROW:
        return None
    row = _AP_ROW[prefix]                       # +anterior
    if idx == '' or idx == 'Z':
        col = 0.0
    else:
        col = _LR_COL.get(idx[-1])
        if col is None:
            return None
    # Flat topographic coords: X = col (right), Y = row (anterior)
    ecc = np.hypot(col, row)                     # grid distance from Cz
    theta = np.deg2rad(ecc * _ECC_STEP_DEG)      # polar angle from vertex
    phi = np.arctan2(row, col)                   # azimuth (0 = right, +anterior)
    x = np.sin(theta) * np.cos(phi)   # right
    y = np.sin(theta) * np.sin(phi)   # anterior
    z = np.cos(theta)                 # superior
    return (float(x), float(y), float(z))


def build_distance_adjacency(channel_names: List[str], sigma: float = 0.5) -> np.ndarray:
    """
    Build a fixed adjacency matrix from real 10-20 scalp geometry.

    Edge weight = exp(-d_ij^2 / (2 * sigma^2)) where d_ij is the Euclidean
    distance between electrodes projected on the unit sphere. Falls back to
    an electrode-index proxy only for labels not found in the 10-20 layout.
    """
    n = len(channel_names)
    pos = [_electrode_3d_position(c) for c in channel_names]
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pos[i] is not None and pos[j] is not None:
                d = np.linalg.norm(np.array(pos[i]) - np.array(pos[j]))
                W[i, j] = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
            else:
                # Fallback for unknown labels: mild index-based decay
                W[i, j] = np.exp(-abs(i - j) / 5.0)
    return W


def _default_channel_order(n_channels: int) -> Optional[List[str]]:
    """Resolve a canonical channel order from the channel count."""
    if n_channels == 62:
        return SEED_62CH
    if n_channels == 32:
        return DEAP_32CH
    return None


# ============================================================================
# Part 1: Data Preprocessing
# ============================================================================

class EEGPreprocessor:
    """
    EEG data preprocessing pipeline for emotion recognition.
    Handles bandpass filtering, frequency band decomposition,
    and differential entropy feature extraction.
    """
    
    # Standard frequency bands
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Electrode grouping for brain region assignment (62-channel ESI system)
    ELECTRODE_REGIONS = {
        'frontal': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 
                     'FZ', 'F2', 'F4', 'F6', 'F8'],
        'temporal': ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                      'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                      'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
        'parietal': ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
        'occipital': ['PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 
                       'CB1', 'O1', 'OZ', 'O2', 'CB2'],
        'central': ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6']  # overlap with temporal
    }
    
    def __init__(self, fs: int = 200, window_size: float = 1.0):
        """
        Args:
            fs: Sampling frequency in Hz
            window_size: STFT window size in seconds
        """
        self.fs = fs
        self.window_size = window_size
        self.nperseg = int(fs * window_size)
        
    def bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, 
                        order: int = 4) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def extract_de_features(self, data: np.ndarray, bands: Optional[Dict] = None) -> np.ndarray:
        """
        Extract Differential Entropy (DE) features from EEG data.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            bands: Dict of frequency bands, defaults to BANDS
            
        Returns:
            DE features of shape (n_channels, n_bands, n_segments)
        """
        if bands is None:
            bands = self.BANDS
            
        n_channels, n_samples = data.shape
        n_bands = len(bands)
        
        # STFT decomposition
        f, t, Zxx = stft(data, fs=self.fs, nperseg=self.nperseg, 
                          noverlap=self.nperseg // 2, axis=-1)
        
        n_segments = Zxx.shape[-1]
        de_features = np.zeros((n_channels, n_bands, n_segments))
        
        for b_idx, (band_name, (low, high)) in enumerate(bands.items()):
            # Find frequency indices within the band
            freq_mask = (f >= low) & (f <= high)
            band_power = np.abs(Zxx[:, freq_mask, :]) ** 2
            
            # Compute DE: DE = 0.5 * log(2 * pi * e * sigma^2)
            # For each channel and segment
            for ch in range(n_channels):
                for seg in range(n_segments):
                    sigma_sq = np.mean(band_power[ch, :, seg]) + 1e-10
                    de_features[ch, b_idx, seg] = 0.5 * np.log(
                        2 * np.pi * np.e * sigma_sq
                    )
        
        return de_features
    
    def compute_plv(self, data: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """
        Compute Phase Locking Value (PLV) between all channel pairs.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            band: Frequency band (low, high)

        Returns:
            PLV matrix of shape (n_channels, n_channels)
        """
        n_channels, n_samples = data.shape

        # Filter to specific band
        filtered = self.bandpass_filter(data, band[0], band[1])

        # Compute analytic signal using scipy Hilbert transform
        analytic = hilbert(filtered, axis=-1)

        # Extract instantaneous phase
        phase = np.angle(analytic)

        # Compute PLV matrix
        plv = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                phase_diff = phase[i] - phase[j]
                plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))

        return plv
    
    def build_assignment_matrix(self, channel_names: List[str]) -> torch.Tensor:
        """
        Build soft assignment matrix from electrodes to brain regions.
        
        Args:
            channel_names: List of electrode channel names
            
        Returns:
            Assignment matrix of shape (n_channels, n_regions)
        """
        n_channels = len(channel_names)
        n_regions = len(self.ELECTRODE_REGIONS)
        
        M = np.zeros((n_channels, n_regions))
        region_names = list(self.ELECTRODE_REGIONS.keys())
        
        for i, ch_name in enumerate(channel_names):
            ch_u = ch_name.upper()
            for j, region_name in enumerate(region_names):
                targets = [c.upper() for c in self.ELECTRODE_REGIONS[region_name]]
                if ch_u in targets:
                    M[i, j] = 1.0
        
        # Handle channels in multiple regions by normalization
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / (row_sums + 1e-10)
        
        return torch.tensor(M, dtype=torch.float32)
    
    def baseline_correct(self, data: np.ndarray, baseline_samples: int = 384) -> np.ndarray:
        """
        Baseline correction: subtract mean of baseline period from each channel.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            baseline_samples: Number of samples in baseline period

        Returns:
            Baseline-corrected data
        """
        baseline = data[:, :baseline_samples].mean(axis=1, keepdims=True)
        return data - baseline

    def process_trial(self, data: np.ndarray, channel_names: List[str],
                      compute_plv_flag: bool = True, do_baseline: bool = False,
                      baseline_samples: int = 384) -> Dict:
        """
        Process a single EEG trial.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            channel_names: List of channel names
            compute_plv_flag: Whether to compute PLV
            
        Returns:
            Dict containing DE features, PLV matrix, and assignment matrix
        """
        # Optional baseline correction (for DEAP)
        if do_baseline:
            data = self.baseline_correct(data, baseline_samples)

        # Extract DE features
        de_features = self.extract_de_features(data)
        # Shape: (n_channels, n_bands, n_segments)
        
        # Flatten band and segment dimensions for GNN input
        n_channels, n_bands, n_segments = de_features.shape
        de_flat = de_features.transpose(0, 2, 1).reshape(n_channels, n_segments * n_bands)
        
        result = {
            'de_features': torch.tensor(de_flat, dtype=torch.float32),
            'de_3d': torch.tensor(de_features, dtype=torch.float32),
            'assignment_matrix': self.build_assignment_matrix(channel_names)
        }
        
        if compute_plv_flag:
            # Compute PLV averaged across all frequency bands (functional
            # connectivity is band-specific; averaging avoids over-relying on
            # a single band and matches the multi-band DE representation).
            plv_bands = [self.compute_plv(data, band) for band in self.BANDS.values()]
            plv_avg = np.mean(plv_bands, axis=0)
            result['plv'] = torch.tensor(plv_avg, dtype=torch.float32)
        
        return result


# ============================================================================
# Part 2: Model Components
# ============================================================================

class AdaptiveAdjacencyMatrix(nn.Module):
    """
    Adaptive adjacency matrix that dynamically fuses learnable structure
    with PLV-based functional connectivity.

    A_adapt = alpha * W_learn + (1 - alpha) * PLV(t)

    When use_adaptive_adj=False, falls back to fixed distance-based adjacency.
    """

    def __init__(self, n_channels: int, hidden_dim: int = 64,
                 use_adaptive_adj: bool = True, alpha_init: float = 0.5,
                 channel_names: Optional[List[str]] = None):
        super().__init__()
        self.n_channels = n_channels
        self.use_adaptive_adj = use_adaptive_adj
        # Channel order for physically-meaningful fixed adjacency
        self.channel_names = channel_names if channel_names is not None \
            else _default_channel_order(n_channels)

        # Learnable adjacency matrix (initialized as upper triangular for symmetry)
        self.W_learn = nn.Parameter(torch.randn(n_channels, n_channels) * 0.1)

        # Fixed distance-based adjacency (fallback)
        self.register_buffer('W_fixed', torch.eye(n_channels))
        self._init_fixed = False  # Will be initialized on first forward

        # Alpha prediction network (input dim = n_channels, adaptive to input)
        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Initialize alpha_net bias so initial output ≈ alpha_init
        # sigmoid(bias) = alpha_init → bias = logit(alpha_init)
        import math
        if alpha_init <= 0.001:
            bias_val = -10.0
        elif alpha_init >= 0.999:
            bias_val = 10.0
        else:
            bias_val = math.log(alpha_init / (1 - alpha_init))
        with torch.no_grad():
            self.alpha_net[-2].bias.fill_(bias_val)

    def _build_fixed_adjacency(self, n_channels: int, device: torch.device):
        """Build fixed adjacency from real 10-20 scalp geometry."""
        if self._init_fixed:
            return
        names = self.channel_names if self.channel_names is not None \
            else _default_channel_order(n_channels)
        if names is not None and len(names) == n_channels:
            # Physically-meaningful distances from projected 3D positions
            W = torch.from_numpy(build_distance_adjacency(names)).float()
        else:
            # Fallback: electrode-index proxy (only if geometry unavailable)
            W = torch.zeros(n_channels, n_channels)
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        W[i, j] = float(np.exp(-abs(i - j) / 5.0))
        self.W_fixed = W.to(device)
        self._init_fixed = True

    def forward(self, x: torch.Tensor, plv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (batch, n_channels, hidden_dim)
            plv: PLV matrix of shape (batch, n_channels, n_channels) or None

        Returns:
            Adaptive adjacency matrix of shape (batch, n_channels, n_channels)
        """
        batch_size = x.shape[0]
        device = x.device

        if not self.use_adaptive_adj:
            # Fixed distance-based adjacency
            self._build_fixed_adjacency(self.n_channels, device)
            A = self.W_fixed.unsqueeze(0).expand(batch_size, -1, -1)
            # Normalize
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-10)
            return A

        # Symmetrize learnable matrix
        W_sym = 0.5 * (self.W_learn + self.W_learn.T)

        # Predict alpha: pool across channels, then through MLP
        x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        alpha = self.alpha_net(x_pooled)  # (batch, 1)
        alpha = alpha.view(batch_size, 1, 1)

        # Expand learnable matrix to batch
        A_learn = W_sym.unsqueeze(0).expand(batch_size, -1, -1)
        
        if plv is not None:
            # Fuse learnable and PLV-based adjacency
            A_adapt = alpha * A_learn + (1 - alpha) * plv
        else:
            A_adapt = A_learn
            
        # Apply soft threshold for sparsity
        A_adapt = F.softplus(A_adapt - 0.1)
        
        # Normalize
        A_adapt = A_adapt / (A_adapt.sum(dim=-1, keepdim=True) + 1e-10)
        
        return A_adapt


class MultiScaleGraphConstruction(nn.Module):
    """
    Construct graphs at three scales: electrode-level, brain-region-level, and global.
    """
    
    def __init__(self, n_channels: int, n_regions: int = 5, hidden_dim: int = 64,
                 use_adaptive_adj: bool = True, alpha_init: float = 0.5,
                 channel_names: Optional[List[str]] = None):
        super().__init__()
        self.n_channels = n_channels
        self.n_regions = n_regions

        # Adaptive adjacency for electrode-level
        self.adaptive_adj = AdaptiveAdjacencyMatrix(
            n_channels, hidden_dim, use_adaptive_adj=use_adaptive_adj,
            alpha_init=alpha_init, channel_names=channel_names
        )

        # Region assignment (learnable refinement from hidden_dim features)
        self.region_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_regions)
        )
        
    def forward(self, x: torch.Tensor, M: torch.Tensor, 
                plv: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Node features (batch, n_channels, hidden_dim)
            M: Assignment matrix (n_channels, n_regions)
            plv: PLV matrix (batch, n_channels, n_channels)
            
        Returns:
            Dict with graphs at three scales
        """
        batch_size, n_channels, n_features = x.shape
        
        # Electrode-level adjacency
        A_elec = self.adaptive_adj(x, plv)  # (batch, n_channels, n_channels)
        
        # Soft region assignment: learn from features
        # x: (batch, n_channels, hidden_dim) → M_soft: (batch, n_channels, n_regions)
        M_soft = F.softmax(self.region_refine(x), dim=-1)
        
        # Region features via weighted pooling
        x_region = torch.bmm(M_soft.transpose(1, 2), x)  # (batch, n_regions, n_features)
        
        # Region adjacency via aggregation
        A_region = torch.bmm(torch.bmm(M_soft.transpose(1, 2), A_elec), M_soft)
        # (batch, n_regions, n_regions)
        
        # Global-level: average pooling
        x_global = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        A_global = torch.ones(batch_size, 1, 1, device=x.device)
        
        return {
            'elec': {'x': x, 'A': A_elec},
            'region': {'x': x_region, 'A': A_region},
            'global': {'x': x_global, 'A': A_global},
            'M': M_soft
        }


class GCNEncoder(nn.Module):
    """Multi-layer GCN encoder with residual connections."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Middle layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Last layer
        self.convs.append(GCNConv(hidden_dim, out_dim))
        self.norms.append(nn.LayerNorm(out_dim))
        
        # Projection for residual when dimensions don't match
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features (n_nodes, in_dim)
            edge_index: Graph connectivity (2, n_edges)
            edge_weight: Edge weights (n_edges,) or None
        """
        identity = self.res_proj(x)
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Residual connection
        x = x + identity
        return x


class CrossScaleMessagePassing(nn.Module):
    """
    Bidirectional cross-scale message passing.
    
    Bottom-up: electrode → region → global (coarsening)
    Top-down: global → region → electrode (refinement)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lambda_up = nn.Parameter(torch.tensor(0.1))
        self.lambda_down = nn.Parameter(torch.tensor(0.1))
        
        # Transformation for cross-scale alignment
        self.up_proj = nn.Linear(hidden_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, h_elec: torch.Tensor, h_region: torch.Tensor, 
                h_global: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            h_elec: (batch, n_channels, hidden_dim)
            h_region: (batch, n_regions, hidden_dim)
            h_global: (batch, 1, hidden_dim)
            M: Assignment matrix (batch, n_channels, n_regions)
            
        Returns:
            Updated h_elec, h_region, h_global
        """
        batch_size = h_elec.shape[0]
        
        # Bottom-up: electrode → region → global
        elec_to_region = torch.bmm(M.transpose(1, 2), self.up_proj(h_elec))
        h_region = h_region + self.lambda_up * elec_to_region
        
        region_to_global = h_region.mean(dim=1, keepdim=True)
        h_global = h_global + self.lambda_up * region_to_global
        
        # Top-down: global → region → electrode
        global_to_region = self.down_proj(h_global).expand(-1, h_region.shape[1], -1)
        h_region = h_region + self.lambda_down * global_to_region
        
        region_to_elec = torch.bmm(M, self.down_proj(h_region))
        h_elec = h_elec + self.lambda_down * region_to_elec
        
        return h_elec, h_region, h_global


class ScaleAwareAttention(nn.Module):
    """Attention-based fusion of multi-scale representations."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.ModuleDict({
            'elec': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            ),
            'region': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            ),
            'global': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )
        })
        
    def forward(self, z_elec: torch.Tensor, z_region: torch.Tensor, 
                z_global: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_elec: Pooled electrode features (batch, hidden_dim)
            z_region: Pooled region features (batch, hidden_dim)
            z_global: Global features (batch, hidden_dim)
            
        Returns:
            Fused representation (batch, hidden_dim)
        """
        # Compute attention scores
        scores = []
        for key, z in [('elec', z_elec), ('region', z_region), ('global', z_global)]:
            scores.append(self.attention[key](z))
        
        scores = torch.cat(scores, dim=-1)  # (batch, 3)
        alpha = F.softmax(scores, dim=-1)  # (batch, 3)
        
        # Weighted fusion
        z_fused = (alpha[:, 0:1] * z_elec + 
                   alpha[:, 1:2] * z_region + 
                   alpha[:, 2:3] * z_global)
        
        return z_fused, alpha


# ============================================================================
# Part 3: AM-DGCN Full Model
# ============================================================================

class AMDGCN(nn.Module):
    """
    Adaptive Multi-Scale Dynamic Graph Convolutional Network
    for EEG-based Emotion Recognition.
    """
    
    def __init__(self, n_channels: int = 62, n_regions: int = 5, 
                 n_classes: int = 3, n_bands: int = 5,
                 hidden_dim: int = 64, dropout: float = 0.5,
                 # Ablation control flags
                 use_multi_scale: bool = True,
                 use_adaptive_adj: bool = True,
                 use_plv: bool = True,
                 use_cross_scale: bool = True,
                 use_scale_attention: bool = True,
                 use_linear_shortcut: bool = True,
                 use_gcn_pathway: bool = True,
                 # Hyperparameter analysis
                 alpha_init: float = 0.5,
                 n_gcn_layers: int = 2,
                 channel_names: Optional[List[str]] = None):
        super().__init__()
        self.n_channels = n_channels
        self.channel_names = channel_names if channel_names is not None \
            else _default_channel_order(n_channels)
        self.n_regions = n_regions if use_multi_scale else 1  # 1 = electrode only
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Store flags for forward pass
        self.use_multi_scale = use_multi_scale
        self.use_adaptive_adj = use_adaptive_adj
        self.use_plv = use_plv
        self.use_cross_scale = use_cross_scale
        self.use_scale_attention = use_scale_attention
        self.use_linear_shortcut = use_linear_shortcut
        self.use_gcn_pathway = use_gcn_pathway
        self.alpha_init = alpha_init
        self.n_gcn_layers = n_gcn_layers
        
        # Input projection
        self.input_norm = nn.BatchNorm1d(n_channels * n_bands)  # 逐特征标准化(类StandardScaler, 保留绝对幅值)
        self.input_proj = nn.Linear(n_bands, hidden_dim)
        # Linear path: flattened normalized input → hidden_dim (captures cross-channel linear interactions)
        self.linear_path = nn.Linear(n_channels * n_bands, hidden_dim)
        
        # Multi-scale graph construction / GCN encoders (skipped entirely for the
        # pure-linear variant, so its parameter count reflects only shortcut+head)
        if use_gcn_pathway:
            self.graph_constructor = MultiScaleGraphConstruction(
                n_channels, self.n_regions, hidden_dim,
                use_adaptive_adj=use_adaptive_adj,
                alpha_init=alpha_init,
                channel_names=self.channel_names
            )

            # Multi-scale GCN encoders
            self.encoder_elec = GCNEncoder(hidden_dim, hidden_dim, hidden_dim, n_layers=n_gcn_layers, dropout=dropout)
            if use_multi_scale:
                self.encoder_region = GCNEncoder(hidden_dim, hidden_dim, hidden_dim, n_layers=n_gcn_layers, dropout=dropout)
                self.encoder_global = GCNEncoder(hidden_dim, hidden_dim, hidden_dim, n_layers=1, dropout=dropout)

            # Cross-scale message passing
            if use_cross_scale:
                self.cross_scale = CrossScaleMessagePassing(hidden_dim)

            # Scale-aware attention fusion
            if use_scale_attention:
                self.scale_attention = ScaleAwareAttention(hidden_dim)
        
        # Classification head
        # When use_scale_attention=False and use_multi_scale=True, we sum 3 scales
        # The output dim stays hidden_dim regardless (all fusion methods produce hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def _dense_to_sparse(self, A: torch.Tensor, threshold: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert dense adjacency matrix to sparse edge_index and edge_weight."""
        batch_size, n_nodes, _ = A.shape
        
        edge_indices = []
        edge_weights = []
        
        for b in range(batch_size):
            mask = A[b] > threshold
            edge_index = torch.nonzero(mask, as_tuple=False).t()
            
            # Add self-loops
            self_loops = torch.arange(n_nodes, device=A.device)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            
            # Extract weights
            weights = A[b][mask]
            self_weights = torch.ones(n_nodes, device=A.device) * 0.1
            edge_weight = torch.cat([weights, self_weights], dim=0)
            
            # Offset for batching
            edge_index = edge_index + b * n_nodes
            
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
        
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights, dim=0)
        
        return edge_index, edge_weight
    
    def forward(self, x: torch.Tensor, M: torch.Tensor, 
                plv: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: EEG DE features (batch, n_channels, n_segments * n_bands)
            M: Electrode-to-region assignment matrix (n_channels, n_regions)
            plv: PLV matrix (batch, n_channels, n_channels) or None

        Returns:
            Dict with logits, predictions, and intermediate representations
        """
        batch_size = x.shape[0]

        # Reshape to separate bands and segments
        x = x.view(batch_size, self.n_channels, -1, 5)  # (batch, n_channels, n_segments, n_bands)
        x = x.mean(dim=2)  # Average over time segments → (batch, n_channels, n_bands)

        # Input standardization (per channel×band, like StandardScaler) + projection
        orig_shape = x.shape  # (batch, n_channels, n_bands)
        x_flat = x.reshape(batch_size, -1)  # (batch, n_channels * n_bands)
        x_norm = self.input_norm(x_flat).reshape(orig_shape)  # normalized (batch, n_channels, n_bands)
        # Linear path: captures cross-channel linear interactions (matches linear probe)
        z_linear = self.linear_path(x_norm.reshape(batch_size, -1))  # (batch, hidden_dim)

        # Pure-linear variant (no GCN pathway): shortcut + MLP head only. Isolates
        # whether the electrode GCN adds any accuracy beyond the shortcut+head.
        if not self.use_gcn_pathway:
            z_fused = z_linear
            alpha = torch.ones(batch_size, 3, device=x.device) / 3.0
            logits = self.classifier(z_fused)
            predictions = torch.argmax(logits, dim=-1)
            return {
                'logits': logits, 'predictions': predictions, 'features': z_fused,
                'alpha': alpha, 'h_elec': z_linear, 'h_region': z_linear,
                'h_global': z_linear,
                'A_elec': torch.zeros(batch_size, self.n_channels, self.n_channels, device=x.device)
            }

        x = self.input_proj(x_norm)  # (batch, n_channels, hidden_dim) for GCN

        # Override PLV if disabled
        if not self.use_plv:
            plv = None

        # Multi-scale graph construction
        graphs = self.graph_constructor(x, M, plv)

        # --- Electrode-level encoding (always on) ---
        edge_idx_elec, edge_w_elec = self._dense_to_sparse(graphs['elec']['A'])
        h_elec = graphs['elec']['x'].view(batch_size * self.n_channels, self.hidden_dim)
        h_elec = self.encoder_elec(h_elec, edge_idx_elec, edge_w_elec)
        h_elec = h_elec.view(batch_size, self.n_channels, self.hidden_dim)

        if self.use_multi_scale:
            # --- Region-level encoding ---
            edge_idx_region, edge_w_region = self._dense_to_sparse(graphs['region']['A'])
            h_region = graphs['region']['x'].view(batch_size * self.n_regions, self.hidden_dim)
            h_region = self.encoder_region(h_region, edge_idx_region, edge_w_region)
            h_region = h_region.view(batch_size, self.n_regions, self.hidden_dim)

            # --- Global-level encoding ---
            edge_idx_global, edge_w_global = self._dense_to_sparse(graphs['global']['A'])
            h_global = graphs['global']['x'].view(batch_size * 1, self.hidden_dim)
            h_global = self.encoder_global(h_global, edge_idx_global, edge_w_global)
            h_global = h_global.view(batch_size, 1, self.hidden_dim)

            # Cross-scale message passing
            if self.use_cross_scale:
                h_elec, h_region, h_global = self.cross_scale(
                    h_elec, h_region, h_global, graphs['M']
                )

            # Pool
            z_elec = h_elec.mean(dim=1)
            z_region = h_region.mean(dim=1)
            z_global = h_global.squeeze(1)

            if self.use_scale_attention:
                z_fused, alpha = self.scale_attention(z_elec, z_region, z_global)
            else:
                # Equal-weight fusion (fallback for ablation)
                z_fused = (z_elec + z_region + z_global) / 3.0
                alpha = torch.ones(batch_size, 3, device=x.device) / 3.0
        else:
            # Single-scale: electrode only
            z_elec = h_elec.mean(dim=1)
            z_fused = z_elec
            z_region = z_elec  # placeholder
            z_global = z_elec  # placeholder
            alpha = torch.ones(batch_size, 3, device=x.device) / 3.0

        # Linear path: add flattened linear features (captures cross-channel interactions)
        if self.use_linear_shortcut:
            z_fused = z_fused + z_linear

        # Classification
        logits = self.classifier(z_fused)
        predictions = torch.argmax(logits, dim=-1)

        return {
            'logits': logits,
            'predictions': predictions,
            'features': z_fused,
            'alpha': alpha,
            'h_elec': z_elec,
            'h_region': z_region,
            'h_global': z_global,
            'A_elec': graphs['elec']['A']
        }


# ============================================================================
# Part 4: Training and Evaluation
# ============================================================================

class Trainer:
    """Training and evaluation utilities for AM-DGCN."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Store model kwargs for re-initialization in CV
        self._model_kwargs = {
            'n_channels': model.n_channels,
            'n_regions': model.n_regions,
            'n_classes': model.n_classes,
            'hidden_dim': model.hidden_dim,
            'n_bands': getattr(model, 'n_bands', 5),
            'dropout': getattr(model, 'dropout', 0.5),
            'use_multi_scale': getattr(model, 'use_multi_scale', True),
            'use_adaptive_adj': getattr(model, 'use_adaptive_adj', True),
            'use_plv': getattr(model, 'use_plv', True),
            'use_cross_scale': getattr(model, 'use_cross_scale', True),
            'use_scale_attention': getattr(model, 'use_scale_attention', True),
            'use_linear_shortcut': getattr(model, 'use_linear_shortcut', True),
            'use_gcn_pathway': getattr(model, 'use_gcn_pathway', True),
            'alpha_init': getattr(model, 'alpha_init', 0.5),
            'n_gcn_layers': getattr(model, 'n_gcn_layers', 2),
            'channel_names': getattr(model, 'channel_names', None),
        }
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            M = batch['assignment_matrix'].to(self.device)
            plv = batch['plv'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(x, M, plv)
            loss = criterion(outputs['logits'], y)
            
            # L1 regularization on adjacency matrix (reduced from 0.01 to prevent over-constraining)
            l1_reg = 0.001 * outputs['A_elec'].abs().mean()
            loss = loss + l1_reg
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            correct += (outputs['predictions'] == y).sum().item()
            total += y.size(0)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            M = batch['assignment_matrix'].to(self.device)
            plv = batch['plv'].to(self.device)
            
            outputs = self.model(x, M, plv)
            loss = criterion(outputs['logits'], y)
            
            total_loss += loss.item() * x.size(0)
            correct += (outputs['predictions'] == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Guard: empty loader
        if total == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0}
        
        # Compute F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'f1_score': f1
        }
    
    def subject_dependent_cv(self, dataset, n_folds: int = 5, epochs: int = 150,
                             lr: float = 0.001, batch_size: int = 64,
                             patience: int = 50, checkpoint_path: Optional[str] = None,
                             resume: bool = True) -> Dict[str, List[float]]:
        """
        Subject-dependent k-fold cross-validation.

        Returns:
            Dict with accuracy and f1 lists for each fold

        If ``checkpoint_path`` is given, per-fold results are written
        immediately after each fold (and re-loaded on restart when
        ``resume=True``), so a crash does not lose accumulated progress.
        """
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        accuracies = []
        f1_scores = []

        # Resume from a previous (possibly partial) checkpoint.
        done = 0
        if checkpoint_path and resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    prev = json.load(f)
                accuracies = list(prev.get('accuracy', []))
                f1_scores = list(prev.get('f1_score', []))
                done = len(accuracies)
                print(f"[resume] loaded {done} completed fold(s) from {checkpoint_path}")
            except Exception as e:
                print(f"[resume] failed to load checkpoint ({e}); starting fresh")

        # Get all trial indices
        n_trials = len(dataset)
        indices = np.arange(n_trials)

        # Materialize splits so we can skip already-completed folds.
        all_splits = list(kfold.split(indices))

        for fold, (train_idx, test_idx) in enumerate(all_splits):
            if fold < done:
                print(f"Fold {fold + 1}/{n_folds} [skipped, already done]")
                continue
            print(f"Fold {fold + 1}/{n_folds}")

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Reinitialize model for each fold
            model = AMDGCN(**self._model_kwargs).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )

            best_acc = 0.0
            best_f1 = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                self.model = model  # Update trainer's model reference
                train_metrics = self.train_epoch(train_loader, optimizer, criterion)
                val_metrics = self.evaluate(test_loader, criterion)

                scheduler.step(val_metrics['accuracy'])

                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy']
                    best_f1 = val_metrics['f1_score']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            accuracies.append(best_acc)
            f1_scores.append(best_f1)
            print(f"  Fold {fold + 1}: Acc={best_acc:.4f}, F1={best_f1:.4f}")

            if checkpoint_path:
                with open(checkpoint_path, 'w') as f:
                    json.dump({'accuracy': accuracies, 'f1_score': f1_scores}, f, indent=2)

        return {
            'accuracy': accuracies,
            'f1_score': f1_scores,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }
    
    def subject_independent_loso(self, dataset, n_subjects: int, epochs: int = 150,
                                  lr: float = 0.001, batch_size: int = 64,
                                  patience: int = 50, checkpoint_path: Optional[str] = None,
                                  resume: bool = True) -> Dict[str, List[float]]:
        """
        Subject-independent Leave-One-Subject-Out cross-validation.

        Args:
            dataset: Full dataset with subject IDs
            n_subjects: Number of subjects

        Returns:
            Dict with per-subject accuracy and aggregate metrics

        If ``checkpoint_path`` is given, per-subject results are written
        immediately after each subject (and re-loaded on restart when
        ``resume=True``), so a crash does not lose accumulated progress.
        """
        accuracies = []
        f1_scores = []

        done = 0
        if checkpoint_path and resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    prev = json.load(f)
                accuracies = list(prev.get('accuracy', []))
                f1_scores = list(prev.get('f1_score', []))
                done = len(accuracies)
                print(f"[resume] loaded {done} completed subject(s) from {checkpoint_path}")
            except Exception as e:
                print(f"[resume] failed to load checkpoint ({e}); starting fresh")

        unique_subjects = sorted(set(dataset.subject_ids))
        for si_idx, test_subject in enumerate(unique_subjects):
            if si_idx < done:
                print(f"Test Subject {si_idx + 1}/{len(unique_subjects)} (id={test_subject}) [skipped, already done]")
                continue
            print(f"Test Subject {si_idx + 1}/{len(unique_subjects)} (id={test_subject})")

            # Split by subject
            train_indices = [i for i, s in enumerate(dataset.subject_ids) if s != test_subject]
            test_indices = [i for i, s in enumerate(dataset.subject_ids) if s == test_subject]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Re-use the exact model kwargs (incl. ablation flags) so that
            # variant models are preserved across LOSO folds.
            model = AMDGCN(**self._model_kwargs).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )

            best_acc = 0.0
            best_f1 = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                self.model = model
                train_metrics = self.train_epoch(train_loader, optimizer, criterion)
                val_metrics = self.evaluate(test_loader, criterion)

                scheduler.step(val_metrics['accuracy'])

                if val_metrics['accuracy'] > best_acc:
                    best_acc = val_metrics['accuracy']
                    best_f1 = val_metrics['f1_score']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            accuracies.append(best_acc)
            f1_scores.append(best_f1)
            print(f"  Subject {test_subject}: Acc={best_acc:.4f}, F1={best_f1:.4f}")

            if checkpoint_path:
                with open(checkpoint_path, 'w') as f:
                    json.dump({'accuracy': accuracies, 'f1_score': f1_scores}, f, indent=2)

        return {
            'accuracy': accuracies,
            'f1_score': f1_scores,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores)
        }


# ============================================================================
# Part 5: Main Execution Example
# ============================================================================

def main():
    """
    Example usage of AM-DGCN.
    
    This is a skeleton that demonstrates the full pipeline.
    Actual execution requires downloading SEED/DEAP datasets.
    """
    # Configuration
    config = {
        'n_channels': 62,      # SEED: 62 channels
        'n_regions': 5,        # frontal, temporal, parietal, occipital, central
        'n_classes': 3,        # SEED: positive, neutral, negative
        'n_bands': 5,          # delta, theta, alpha, beta, gamma
        'hidden_dim': 64,
        'dropout': 0.5,
        'batch_size': 64,
        'epochs': 200,
        'lr': 0.001,
    }
    
    print("=" * 60)
    print("AM-DGCN: Adaptive Multi-Scale Dynamic Graph Convolutional Network")
    print("for EEG-Based Emotion Recognition")
    print("=" * 60)
    print(f"\nConfiguration: {config}")
    
    # Initialize model
    model = AMDGCN(
        n_channels=config['n_channels'],
        n_regions=config['n_regions'],
        n_classes=config['n_classes'],
        n_bands=config['n_bands'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(model)
    
    print("\n" + "=" * 60)
    print("Model ready for training.")
    print("To run experiments, please:")
    print("  1. Download SEED dataset from: https://bcmi.sjtu.edu.cn/home/seed/")
    print("  2. Download DEAP dataset from: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
    print("  3. Preprocess data using EEGPreprocessor class")
    print("  4. Create PyTorch Dataset with processed features")
    print("  5. Run trainer.subject_dependent_cv() or trainer.subject_independent_loso()")
    print("=" * 60)
    
    return model, trainer


if __name__ == '__main__':
    model, trainer = main()
