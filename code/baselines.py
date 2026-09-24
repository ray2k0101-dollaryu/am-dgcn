"""Baseline GNN models for the same-protocol comparison requested by the
pre-submission review (item A2: 're-run at least two representative GNN
baselines under the identical DE-feature / LOSO protocol').

Why this file exists
--------------------
The manuscript's Tables 2-7 quote DGCNN, RGNN and others *as published*, under
their own preprocessing, feature and evaluation protocols, and are therefore not
comparable to our numbers.  The review's first question is unavoidable --
"under your protocol, is AM-DGCN better or worse than DGCNN/RGNN?" -- and the
only honest way to answer it is to re-run those models here.

The repository contained NO baseline implementations, so these are written from
scratch.  They are re-implementations from the published architectures, not the
authors' original code, and this must be stated in the manuscript.

Interface contract
------------------
The existing Trainer (am_dgcn_model.Trainer) calls

    outputs = self.model(x, M, plv)
    loss    = criterion(outputs['logits'], y) + 0.001 * outputs['A_elec'].abs().mean()

and Trainer.__init__ additionally reads n_channels / n_regions / n_classes /
hidden_dim / n_bands / dropout off the model.  Every class here therefore:
  * accepts forward(x, M, plv) with
        x    (batch, n_channels, n_segments * n_bands)
        M    (n_channels, n_regions)          -- shared across the batch
        plv  (batch, n_channels, n_channels) or None
  * returns a dict containing 'logits' and 'A_elec' (batch, C, C), so that the
    L1 analysis in Section 4.6 behaves identically for every model
  * exposes the attribute set the Trainer reads.

Seeding is NOT done here.  As with the ablation runners, the single random
stream is applied once at module level in the launcher, so that all models in
this batch share one stream.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DGCNN", "RGNN", "TrainerRGNN", "grad_reverse"]


# ===========================================================================
# Gradient reversal (for NodeDAT's domain-adversarial term)
# ===========================================================================
class _GradReverseFn(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward.

    Inverted gradients are what make the node embeddings *worse* at predicting
    the subject, i.e. what makes them subject-invariant.
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverseFn.apply(x, lambd)


class DGCNN(nn.Module):
    """Dynamical Graph Convolutional Neural Network (Song et al., IEEE TAC 2020).

    Components kept faithful to the published architecture:
      * the four-block 2-D convolutional extractor over (channels x bands)
        with the published widths and kernels,
      * the *dynamical* -- i.e. data-dependent, recomputed per sample --
        adjacency derived from the extracted per-channel features,
      * graph convolution using that adjacency,
      * global pooling over nodes followed by a fully connected classifier.

    Adaptation to the trial-level DE setting (documented in the manuscript):
      The published model consumes (channels, bands, time).  Our samples are
      trial-level DE aggregates, so the segment axis has already been averaged
      and there is no time axis left to pool over.  We therefore drop the
      pooling that the original applies along time and take the mean over the
      residual band axis instead.  A consequence worth stating plainly: the
      adjacency here is computed from each trial's band profile rather than from
      its time course, so this is a band-profile dynamical graph, not the
      time-course dynamical graph of the original.

    Parameter count is reported by the runner and by Appendix A; it is NOT
    matched to AM-DGCN, because shrinking the published widths would make the
    comparison against the literature number less meaningful.  A
    parameter-matched variant can be produced by passing conv_channels.
    """

    # published widths / kernels; all keep the band axis at its input size
    DEFAULT_CONV = (64, 128, 256, 64)
    DEFAULT_KERNEL = ((1, 5), (1, 5), (1, 3), (1, 3))
    DEFAULT_PAD = ((0, 2), (0, 2), (0, 1), (0, 1))

    def __init__(self, n_channels: int = 62, n_regions: int = 5,
                 n_classes: int = 3, n_bands: int = 5,
                 hidden_dim: int = 64, dropout: float = 0.5,
                 conv_channels=DEFAULT_CONV, kernels=DEFAULT_KERNEL,
                 paddings=DEFAULT_PAD, adj_temperature: float = 1.0,
                 **unused_kwargs):
        super().__init__()

        # ---- attributes read by Trainer.__init__ -------------------------
        self.n_channels = n_channels
        self.n_regions = n_regions
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Accepted and ignored: these exist so that a caller may pass the
        # ablation switch names without the class breaking.  It also makes the
        # point explicit that DGCNN has no multi-scale / PLV / shortcut
        # pathway -- there is nothing to ablate.
        self.use_multi_scale = unused_kwargs.get("use_multi_scale", False)
        self.use_adaptive_adj = unused_kwargs.get("use_adaptive_adj", False)
        self.use_plv = unused_kwargs.get("use_plv", False)
        self.use_cross_scale = unused_kwargs.get("use_cross_scale", False)
        self.use_scale_attention = unused_kwargs.get("use_scale_attention", False)
        self.use_linear_shortcut = unused_kwargs.get("use_linear_shortcut", False)
        self.use_gcn_pathway = True
        self.alpha_init = 0.5
        self.n_gcn_layers = 1

        # ---- input normalisation (mirrors AM-DGCN's input_norm) ----------
        # Per-channel-band standardisation of the flattened input, so that both
        # models see identically scaled features.
        self.input_norm = nn.BatchNorm1d(n_channels * n_bands)

        # ---- convolutional extractor over (channels x bands) -------------
        blocks = []
        in_c = 1
        for out_c, k, p in zip(conv_channels, kernels, paddings):
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=k, padding=p),
                nn.BatchNorm2d(out_c),
                nn.ELU(),
            ))
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)
        feat_dim = conv_channels[-1]
        self.feat_dim = feat_dim

        # ---- dynamical adjacency: per-channel feature -> C-dim code ------
        # Linear(feat_dim, C) applied to (batch, C, feat_dim) yields
        # (batch, C, C): the adjacency is a function of the sample's features.
        self.dense = nn.Linear(feat_dim, n_channels)
        self.adj_temperature = adj_temperature

        # ---- graph convolution: A @ (H W) --------------------------------
        self.gconv_w = nn.Linear(feat_dim, hidden_dim)

        # ---- classifier head --------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    # ------------------------------------------------------------------
    def adjacency(self, feat: torch.Tensor) -> torch.Tensor:
        """Data-dependent adjacency.  feat: (batch, C, feat_dim) -> (batch, C, C).

        LeakyReLU followed by row normalisation follows the published model; the
        normalisation keeps the graph-convolution output on a comparable scale
        to AM-DGCN's so the shared L1 term behaves the same way.
        """
        A = F.leaky_relu(self.dense(feat), 0.2)
        A = torch.relu(A)                       # keep the L1 penalty meaningful
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        return A

    def forward(self, x: torch.Tensor, M=None, plv=None):
        batch_size = x.shape[0]

        # (batch, C, S, bands) -> mean over segments -> (batch, C, bands)
        x = x.view(batch_size, self.n_channels, -1, self.n_bands).mean(dim=2)

        # identically scaled input for both models
        x = self.input_norm(x.reshape(batch_size, -1))
        x = x.reshape(batch_size, self.n_channels, self.n_bands)

        # convolutions over the (channels x bands) plane
        h = x.unsqueeze(1)                      # (batch, 1, C, bands)
        for blk in self.blocks:
            h = blk(h)                          # (batch, feat_dim, C, bands)

        feat = h.mean(dim=-1)                   # (batch, feat_dim, C)
        feat = feat.transpose(1, 2)             # (batch, C, feat_dim)

        A = self.adjacency(feat)                # (batch, C, C)

        z = self.gconv_w(feat)                  # (batch, C, hidden_dim)
        z = torch.bmm(A, z)                     # graph convolution
        z = F.elu(z)
        z = z.mean(dim=1)                       # global pooling over nodes

        logits = self.classifier(z)
        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "features": z,
            "A_elec": A,
        }


# ===========================================================================
# RGNN -- Regularized Graph Neural Network (Zhong et al., IEEE TAC 2022)
# ===========================================================================
class RGNN(nn.Module):
    """RGNN re-implemented for the trial-level DE setting.

    Re-implemented components:
      * the regularized GCN backbone: a two-layer graph convolution stack
        (LayerNorm + residual) over a graph whose edge weights are *learned*,
        initialised from an electrode-distance prior;
      * **NodeDAT**: a domain-adversarial regulariser.  A subject classifier is
        attached to the node embeddings through a gradient-reversal layer, so
        the embeddings are pushed to be subject-invariant -- the mechanism the
        published model credits for most of its cross-subject gain.

    NOT re-implemented (stated explicitly rather than silently dropped):
      * **EmotionDL**, the label-distribution regulariser.  It estimates a
        per-subject emotion distribution and is defined over the training
        subjects' labels; under LOSO the held-out subject contributes no labels
        at all, so the term would be computed over 14 of 15 subjects and its
        effect would not be comparable across folds.  We omit it and say so.

    Differences from the published implementation that must be disclosed:
      * dense matrix products instead of the authors' sparse graph ops;
      * the adjacency is shared across the batch (a single learned graph), not
        recomputed per sample -- this is a real architectural difference from
        DGCNN, whose adjacency is data-dependent;
      * no hyperparameter search: the protocol is inherited from AM-DGCN.

    The gradient-reversal strength lambda ramps 0 -> 1 over the first
    `nodedat_ramp` epochs, driven by the runner calling set_epoch(ep) (the
    shared Trainer does not know the epoch number).
    """

    def __init__(self, n_channels: int = 62, n_regions: int = 5,
                 n_classes: int = 3, n_bands: int = 5,
                 hidden_dim: int = 64, dropout: float = 0.5,
                 use_nodedat: bool = True, n_domain_subjects: int = 14,
                 nodedat_weight: float = 1.0, nodedat_ramp: int = 20,
                 adj_sigma: float = 0.5, adj_topk: int = 10,
                 **unused_kwargs):
        super().__init__()

        self.n_channels = n_channels
        self.n_regions = n_regions
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # no multi-scale / PLV / shortcut pathway exists in RGNN
        self.use_multi_scale = False
        self.use_adaptive_adj = True          # it does learn its adjacency
        self.use_plv = False
        self.use_cross_scale = False
        self.use_scale_attention = False
        self.use_linear_shortcut = False
        self.use_gcn_pathway = True
        self.alpha_init = 0.5

        self.input_norm = nn.BatchNorm1d(n_channels * n_bands)
        self.input_proj = nn.Linear(n_bands, hidden_dim)

        # ---- learnable adjacency, distance-prior initialised --------------
        prior, mask = self._distance_prior(n_channels, adj_sigma, adj_topk)
        self.register_buffer("adj_mask", torch.tensor(mask, dtype=torch.bool))
        self.adj_logits = nn.Parameter(
            torch.log(torch.tensor(prior, dtype=torch.float32) + 1e-6))

        # ---- two-layer regularized GCN ------------------------------------
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        # ---- NodeDAT ------------------------------------------------------
        self.use_nodedat = bool(use_nodedat)
        self.nodedat_weight = float(nodedat_weight)
        self.nodedat_ramp = max(1, int(nodedat_ramp))
        self.n_domain_subjects = int(n_domain_subjects)
        self._epoch = 0
        self._lambd = 0.0
        if self.use_nodedat:
            self.domain = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_domain_subjects),
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _distance_prior(n_channels, sigma, topk):
        """Row-normalised distance prior + the k-NN mask that defines the graph."""
        A = None
        try:
            from am_dgcn_model import build_distance_adjacency, _default_channel_order
            names = _default_channel_order(n_channels)
            if names is not None:
                A = np.asarray(build_distance_adjacency(names, sigma=sigma),
                               dtype=np.float64)
        except Exception:
            A = None

        if A is None or A.shape != (n_channels, n_channels) or not np.isfinite(A).all():
            A = np.ones((n_channels, n_channels), dtype=np.float64)
        np.fill_diagonal(A, A.max() if A.max() > 0 else 1.0)   # self-loops
        A = np.clip(A, 0.0, None)

        k = max(1, min(int(topk), n_channels))
        mask = np.zeros_like(A, dtype=bool)
        order = np.argsort(-A, axis=1)[:, :k]
        for i in range(n_channels):
            mask[i, order[i]] = True

        P = np.where(mask, A, 0.0)
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return P, mask

    def dense_adjacency(self):
        """Row-stochastic, learnable, shared across the batch: (C, C)."""
        logits = self.adj_logits.masked_fill(~self.adj_mask, float("-inf"))
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    def set_epoch(self, ep: int):
        """Drive the NodeDAT lambda ramp. Called once per epoch by the runner."""
        self._epoch = int(ep)
        self._lambd = min(1.0, float(ep + 1) / float(self.nodedat_ramp))

    @property
    def lambd(self):
        return self._lambd

    def domain_loss(self, node_emb, subject_ids):
        """NodeDAT adversarial term: node embeddings must not predict the subject.

        `subject_ids` are the *training* subjects remapped to 0..S-1, so the
        classifier has one output per training subject.
        """
        z = grad_reverse(node_emb, self._lambd)
        logits = self.domain(z)                                   # (B, C, S)
        target = subject_ids.view(-1, 1).expand(-1, self.n_channels).reshape(-1)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, M=None, plv=None):
        batch_size = x.shape[0]

        x = x.view(batch_size, self.n_channels, -1, self.n_bands).mean(dim=2)
        x = self.input_norm(x.reshape(batch_size, -1))
        x = x.reshape(batch_size, self.n_channels, self.n_bands)

        h = self.input_proj(x)                       # (B, C, hidden)
        A = self.dense_adjacency()                   # (C, C), shared
        Ab = A.unsqueeze(0).expand(batch_size, -1, -1)

        z = torch.bmm(Ab, self.w1(h))
        z = F.relu(self.norm1(z))
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = torch.bmm(Ab, self.w2(z))
        z = self.norm2(z)
        node = z + h                                 # residual

        g = node.mean(dim=1)                         # graph-level pooling
        logits = self.classifier(g)

        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "features": g,
            "node_embeddings": node,
            "A_elec": Ab,
        }


# ===========================================================================
# Trainer subclass that carries the NodeDAT term
# ===========================================================================
from am_dgcn_model import Trainer as _BaseTrainer  # noqa: E402


class TrainerRGNN(_BaseTrainer):
    """Identical to Trainer except that the NodeDAT adversarial term is added.

    The body of train_epoch is copied from Trainer.train_epoch so that the
    classification loss, the L1 adjacency term, the optimiser stepping and the
    running statistics are bit-for-bit the same procedure.  The ONLY addition is
    the domain loss.  am_dgcn_model.py itself is not modified.

    The extra batch key 'subject_id' is produced by the baseline runner's own
    collate function; the shared collate_fn is left untouched.
    """

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            M = batch["assignment_matrix"].to(self.device)
            plv = batch["plv"].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(x, M, plv)
            loss = criterion(outputs["logits"], y)
            loss = loss + 0.001 * outputs["A_elec"].abs().mean()

            # --- the only addition relative to Trainer.train_epoch ---------
            if getattr(self.model, "use_nodedat", False) and "subject_id" in batch:
                sid = batch["subject_id"].to(self.device)
                loss = loss + self.model.nodedat_weight * self.model.domain_loss(
                    outputs["node_embeddings"], sid)
            # ---------------------------------------------------------------

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (outputs["predictions"] == y).sum().item()
            total += y.size(0)

        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {"loss": total_loss / total, "accuracy": correct / total}
