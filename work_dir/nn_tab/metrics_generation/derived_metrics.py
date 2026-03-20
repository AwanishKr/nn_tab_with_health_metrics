"""
derived_metrics.py
==================
Post-training, CPU-only computation of all 14 DataGenome metrics from the
5 raw signals recorded by TrainingSignalCollector.

Raw signals (inputs)
--------------------
logits_array     : np.ndarray [N, T, C]  float16 → cast to float32 internally
loss_array       : np.ndarray [N, T]     float16/float32
correct_array    : np.ndarray [N, T]     uint8  (1 = correct, 0 = wrong)
grad_norm_array  : np.ndarray [N, K]     float16/float32  (first K epochs only)
embeddings_array : np.ndarray [N, D]     float16/float32
labels           : np.ndarray [N]        int64

Derived metrics (outputs — one scalar or trajectory per sample)
----------------------------------------------------------------
① margin_traj    [N, T]  — logit(true) − max_logit(other) at each epoch
② aum            [N]     — mean(margin, axis=1); negative → likely mislabeled
③ rho            [N]     — MCT τ* / T; NaN if never stably crossed
④ fe             [N]     — forgetting events (correct→wrong transitions)
⑤ confidence_t   [N, T]  — softmax prob of true class at each epoch
⑥ variability    [N]     — std(confidence_t, axis=1)
⑦ el2n           [N]     — mean L2‖softmax − onehot‖ over first e_early epochs
⑧ grand          [N]     — mean(grad_norm_array, axis=1)
⑨ loss_traj      [N, T]  — direct pass-through of loss_array (no computation)
⑩ margin_var     [N]     — var(margin, axis=1); high → oscillating prediction
⑪ knn_density    [N]     — mean_knn_dist / global_mean_dist; <1 = crowded, >1 = isolated
⑫ prototypicality[N]     — cosine_sim(embedding[i], class_centroid[label[i]])
⑭ ensemble_dis   [N]     — fraction of seeds whose final pred ≠ majority; multi-seed

Entry points
------------
derive_all_metrics(...)  → pd.DataFrame, one row per sample, all scalar metrics
Each metric group also has a standalone function for targeted reuse.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_f32(arr: np.ndarray) -> np.ndarray:
    """Cast array to float32 (safe upcasting from float16 before arithmetic)."""
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis. Input: [N, T, C] or [N, C]."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _cosine_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two [N, D] arrays."""
    norm_a = np.linalg.norm(A, axis=1, keepdims=True)
    norm_b = np.linalg.norm(B, axis=1, keepdims=True)
    denom = (norm_a * norm_b).clip(min=1e-10)
    return (A * B).sum(axis=1) / denom.squeeze(axis=1)


# ---------------------------------------------------------------------------
# ① + ⑩  Margin trajectory  →  margin_traj [N, T]  +  margin_var [N]
# ---------------------------------------------------------------------------

def compute_margin_trajectory(
    logits_array: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-epoch margin for every sample.

    margin[i, t] = logit_true[i, t] − max_{c ≠ y_i} logit[i, t, c]
    Positive = correctly ranked; negative = misclassified.

    Returns
    -------
    margin_traj : np.ndarray [N, T]
    margin_var  : np.ndarray [N]   — ⑩  variance across epochs
    """
    logits = _to_f32(logits_array)                      # [N, T, C]
    N, T, C = logits.shape

    # Build fully-advanced index arrays so the result is exactly [N, T]
    idx_n = np.arange(N)[:, None] * np.ones((1, T), dtype=np.intp)   # [N, T]
    idx_t = np.ones((N, 1), dtype=np.intp) * np.arange(T)[None, :]   # [N, T]
    idx_c = np.broadcast_to(labels[:, None], (N, T))                  # [N, T]

    true_logit = logits[idx_n, idx_t, idx_c]            # [N, T] — logit of true class

    # Max logit over non-true classes
    logits_masked = logits.copy()
    logits_masked[idx_n, idx_t, idx_c] = -np.inf
    max_other = logits_masked.max(axis=-1)               # [N, T]

    margin_traj = true_logit - max_other                 # [N, T]
    margin_var  = margin_traj.var(axis=1)                # [N]    ⑩

    return margin_traj, margin_var


# ---------------------------------------------------------------------------
# ②  AUM — Area Under the Margin
# ---------------------------------------------------------------------------

def compute_aum(margin_traj: np.ndarray) -> np.ndarray:
    """
    AUM[i] = mean(margin[i, :])  across all T epochs.
    Negative AUM is the primary indicator of a mislabeled sample.

    Parameters
    ----------
    margin_traj : np.ndarray [N, T]

    Returns
    -------
    aum : np.ndarray [N]
    """
    return margin_traj.mean(axis=1)


# ---------------------------------------------------------------------------
# ③  MCT ρ — Margin Crossover Time (normalised)
# ---------------------------------------------------------------------------

def compute_mct_rho(
    margin_traj: np.ndarray,
    k_mct: int = 5,
) -> np.ndarray:
    """
    For each sample, find the first epoch t where margin stays positive
    for k_mct consecutive epochs (the "stable crossover").

    ρ[i] = τ*[i] / T   — normalised to [0, 1]
    ρ[i] = NaN          — if no stable crossover was ever found.

    Low ρ  → learned early (Easy / Redundant).
    ρ = NaN → never stably learned (Noisy / Boundary candidates).

    Parameters
    ----------
    margin_traj : np.ndarray [N, T]  — pre-computed from compute_margin_trajectory()
    k_mct       : int                — stability window (default 5; use 3 if T < 50)

    Returns
    -------
    rho : np.ndarray [N]   dtype float32; NaN where no stable crossover exists
    """
    N, T = margin_traj.shape
    positive = (margin_traj > 0).astype(np.int8)           # [N, T]  bool→int

    # Rolling sum of k_mct consecutive True values.
    # We use a cumulative sum trick: cumsum[t] - cumsum[t-k] gives sum over window.
    # Window [t, t+k_mct) is stable iff its sum == k_mct.
    cum = np.concatenate([np.zeros((N, 1), dtype=np.int32), positive.cumsum(axis=1)], axis=1)
    # window_sum[i, t] = number of positive epochs in [t, t+k_mct)
    window_sum = cum[:, k_mct:] - cum[:, :T - k_mct + 1]  # [N, T - k_mct + 1]

    stable = (window_sum >= k_mct)                          # [N, T - k_mct + 1]

    # For each sample, find the first column where stable is True
    found = stable.any(axis=1)                              # [N] bool
    # argmax returns 0 if no True found — mask those out with 'found'
    tau_star = np.where(found, stable.argmax(axis=1), -1)   # [N]  epoch index or -1

    rho = np.where(found, tau_star.astype(np.float32) / T, np.nan).astype(np.float32)
    return rho


# ---------------------------------------------------------------------------
# ④  Forgetting Events
# ---------------------------------------------------------------------------

def compute_forgetting_events(correct_array: np.ndarray) -> np.ndarray:
    """
    Count correct→wrong transitions across consecutive epochs.

    fe[i] = Σ_t  [correct[i, t-1] == 1  AND  correct[i, t] == 0]

    fe == 0 is a necessary condition for EASY and REDUNDANT categories.

    Parameters
    ----------
    correct_array : np.ndarray [N, T]  uint8

    Returns
    -------
    fe : np.ndarray [N]  int32
    """
    c = correct_array.astype(np.int8)
    # Transition: was 1 at t-1, is 0 at t  →  (c[:, :-1] - c[:, 1:]) == 1
    transitions = (c[:, :-1] == 1) & (c[:, 1:] == 0)      # [N, T-1]
    return transitions.sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# ⑤ + ⑥  Confidence(t) + Confidence Variability
# ---------------------------------------------------------------------------

def compute_confidence_trajectories(
    logits_array: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Softmax probability of the true class at every epoch, plus two summary scalars.

    confidence_t[i, t]  — softmax P(y=true | x_i) at epoch t    ⑤
    variability[i]       — std(confidence_t[i, :]) across epochs  ⑥
    correctness[i]       — mean(correct_array[i, :]) (time-averaged accuracy)

    Note: correctness here is re-derived from logits for consistency with the
    confidence trajectory.  Pass correct_array directly to compute_forgetting_events()
    to get the fe count.

    Parameters
    ----------
    logits_array : np.ndarray [N, T, C]
    labels       : np.ndarray [N]

    Returns
    -------
    confidence_t : np.ndarray [N, T]
    variability  : np.ndarray [N]
    """
    logits = _to_f32(logits_array)                 # [N, T, C]
    probs  = _softmax(logits)                      # [N, T, C]

    N, T, C = probs.shape
    # Fully-advanced indexing to avoid non-adjacent axis issue → result [N, T]
    idx_n = np.arange(N)[:, None] * np.ones((1, T), dtype=np.intp)
    idx_t = np.ones((N, 1), dtype=np.intp) * np.arange(T)[None, :]
    idx_c = np.broadcast_to(labels[:, None], (N, T))
    confidence_t = probs[idx_n, idx_t, idx_c]     # [N, T] — P(true class) per epoch

    variability = confidence_t.std(axis=1)         # [N]   ⑥

    return confidence_t, variability


# ---------------------------------------------------------------------------
# ⑦  EL2N — Error L2 Norm (early epochs)
# ---------------------------------------------------------------------------

def compute_el2n(
    logits_array: np.ndarray,
    labels: np.ndarray,
    e_early: int = 5,
) -> np.ndarray:
    """
    EL2N[i] = mean over t in [0, e_early) of   L2‖softmax(logits[i,t]) − onehot(y_i)‖

    High EL2N early in training → sample is hard or mislabeled.
    Using the mean over e_early epochs (rather than a single epoch snapshot)
    gives a more robust estimate.

    Parameters
    ----------
    logits_array : np.ndarray [N, T, C]
    labels       : np.ndarray [N]
    e_early      : int  number of early epochs to average over (default 5)

    Returns
    -------
    el2n : np.ndarray [N]  float32
    """
    logits = _to_f32(logits_array)
    N, T, C = logits.shape
    e_early = min(e_early, T)                               # guard against short runs

    early_logits = logits[:, :e_early, :]                   # [N, e_early, C]
    probs        = _softmax(early_logits)                   # [N, e_early, C]

    # One-hot [N, 1, C] — broadcasts over the e_early axis when subtracted from probs
    onehot = np.zeros((N, 1, C), dtype=np.float32)
    onehot[np.arange(N), 0, labels] = 1.0                  # all fully-advanced: result [N]

    error_vecs = probs - onehot                             # [N, e_early, C]
    l2_per_epoch = np.linalg.norm(error_vecs, axis=-1)     # [N, e_early]
    el2n = l2_per_epoch.mean(axis=1)                       # [N]

    return el2n.astype(np.float32)


# ---------------------------------------------------------------------------
# ⑧  GraNd — Gradient Norm score
# ---------------------------------------------------------------------------

def compute_grand(grad_norm_array: np.ndarray) -> np.ndarray:
    """
    GraNd[i] = mean(grad_norm_array[i, :]) across the K early epochs.

    grad_norm_array already stores the per-epoch L2 gradient norm of the last
    linear layer (filled by TrainingSignalCollector.compute_grad_norms).
    Near-zero GraNd combined with early ρ is the defining signature of REDUNDANT.

    Parameters
    ----------
    grad_norm_array : np.ndarray [N, K]

    Returns
    -------
    grand : np.ndarray [N]  float32
    """
    return _to_f32(grad_norm_array).mean(axis=1)


# ---------------------------------------------------------------------------
# ⑨  Per-sample Loss trajectory  (direct pass-through)
# ---------------------------------------------------------------------------

def get_loss_trajectory(loss_array: np.ndarray) -> np.ndarray:
    """
    Metric ⑨ is the loss_array itself — no computation needed.
    Returns a float32 copy for downstream use.

    Parameters
    ----------
    loss_array : np.ndarray [N, T]

    Returns
    -------
    loss_traj : np.ndarray [N, T]  float32
    """
    return _to_f32(loss_array)


# ---------------------------------------------------------------------------
# ⑪  kNN Embedding Density
# ---------------------------------------------------------------------------

def compute_knn_density(
    embeddings_array: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Neighbourhood density in the final penultimate-layer embedding space.

    Two-step computation per sample i:
        1. raw_distances[i]  = mean Euclidean distance to k nearest neighbours (self excluded)
        2. knn_density[i]    = raw_distances[i] / global_mean_distance

    knn_density < 1 → closer than average → high neighbourhood density (REDUNDANT candidate)
    knn_density > 1 → farther than average → low neighbourhood density (RARE candidate)

    Bottom/top 10% thresholding is applied in the categorization layer via percentile rank
    of these normalised values.

    This makes the score directly interpretable as a density percentile:
        knn_density = 1.0  → most crowded (smallest distance, top density)
        knn_density = 0.0  → most isolated (largest distance, bottom density)

    Thresholding is then direct:
        bottom 10% of embedding density → knn_density < 0.10  (RARE)
        top 10% of embedding density    → knn_density > 0.90  (REDUNDANT)

    raw_distances and global_mean_distance are returned for saving to disk.

    Parameters
    ----------
    embeddings_array : np.ndarray [N, D]
    k                : int  number of neighbours (default 10)

    Returns
    -------
    knn_density          : np.ndarray [N]  float32  — normalised distance (raw / global_mean)
    raw_distances        : np.ndarray [N]  float32  — unnormalised mean kNN distance per sample
    global_mean_distance : float           — global mean used for normalisation
    """
    emb = _to_f32(embeddings_array)
    N   = emb.shape[0]

    k_actual = min(k + 1, N)                                # cap for tiny datasets
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", algorithm="auto")
    nn.fit(emb)
    dists, _ = nn.kneighbors(emb)                          # [N, k+1]; col 0 = self (dist=0)
    raw_distances = dists[:, 1:].mean(axis=1)              # [N]; exclude self

    global_mean = float(raw_distances.mean())

    # Step 2: normalise by global mean
    if global_mean == 0:
        knn_density = np.ones(N, dtype=np.float32)
    else:
        knn_density = raw_distances / global_mean           # [N]; <1 = crowded, >1 = isolated

    return knn_density.astype(np.float32), raw_distances.astype(np.float32), global_mean


# ---------------------------------------------------------------------------
# ⑫  Prototypicality
# ---------------------------------------------------------------------------

def compute_prototypicality(
    embeddings_array: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity between each sample's embedding and its class centroid.

    prototypicality[i] = cosine_sim( embedding[i], centroid[labels[i]] )

    → 1.0  sample sits at the class centroid   → strongly prototypical
    → 0.0  sample is orthogonal to the centroid → atypical / outlier

    Parameters
    ----------
    embeddings_array : np.ndarray [N, D]
    labels           : np.ndarray [N]

    Returns
    -------
    prototypicality : np.ndarray [N]  float32  ∈ [0, 1]
    centroids       : np.ndarray [C, D]  float32  (for downstream reuse / inspection)
    """
    emb = _to_f32(embeddings_array)
    unique_classes = np.unique(labels)
    C = unique_classes.max() + 1
    D = emb.shape[1]

    centroids = np.zeros((C, D), dtype=np.float32)
    for c in unique_classes:
        mask = labels == c
        centroids[c] = emb[mask].mean(axis=0)

    # Gather the centroid that corresponds to each sample's label
    class_centroids_per_sample = centroids[labels]          # [N, D]
    prototypicality = _cosine_sim_rows(emb, class_centroids_per_sample)  # [N]

    return prototypicality.astype(np.float32), centroids


# ---------------------------------------------------------------------------
# ⑬  Prediction Depth  (optional — requires intermediate layer activations)
# ---------------------------------------------------------------------------

def compute_prediction_depth(
    intermediate_activations: dict[str, np.ndarray],
    labels: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    Earliest layer at which a kNN probe can correctly classify the sample.

    intermediate_activations: dict mapping a depth label (e.g. '25pct', '50pct',
        '75pct', '100pct') to an activation array [N, D_layer].  Layers are
        processed in the order they appear in the dict (Python 3.7+ insertion order).

    For each sample i, pred_depth[i] = the key of the first layer at which the
    majority vote among k nearest neighbours equals the true label.
    If no layer classifies the sample correctly, pred_depth[i] = NaN.

    Parameters
    ----------
    intermediate_activations : dict[str, np.ndarray [N, D]]
    labels                   : np.ndarray [N]
    k                        : int  number of neighbours for the kNN probe

    Returns
    -------
    pred_depth : np.ndarray [N, dtype=object]  — depth label str or np.nan
    """
    N = labels.shape[0]
    pred_depth = np.full(N, np.nan, dtype=object)

    labels_np = labels.astype(np.int64)

    for depth_label, acts in intermediate_activations.items():
        acts_f = _to_f32(acts)
        k_actual = min(k + 1, N)
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean").fit(acts_f)
        _, neigh_idx = nn.kneighbors(acts_f)                # [N, k+1]

        for i in range(N):
            if pred_depth[i] is not np.nan:                 # already found shallower layer
                continue
            neighbours = neigh_idx[i, 1:]                  # exclude self
            neighbour_labels = labels_np[neighbours]
            majority = np.bincount(neighbour_labels).argmax()
            if majority == labels_np[i]:
                pred_depth[i] = depth_label

    return pred_depth


# ---------------------------------------------------------------------------
# ⑭  Ensemble Disagreement  (multi-seed)
# ---------------------------------------------------------------------------

def compute_ensemble_disagreement(
    per_seed_logits: list[np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fraction of seeds whose final-epoch prediction disagrees with the majority vote.

    Requires the final-epoch logit slice from each seed's logits_array.

    Parameters
    ----------
    per_seed_logits : list of np.ndarray, each shape [N, C]
        logits_array[seed][:, -1, :]  — last epoch logits per seed.
    labels          : np.ndarray [N]

    Returns
    -------
    ensemble_dis  : np.ndarray [N]  float32  ∈ [0, 1]
        Fraction of seeds predicting a class other than the majority vote.
    majority_pred : np.ndarray [N]  int32
        Majority-vote predicted class across seeds.
    seed_agree    : np.ndarray [N]  bool
        True if ALL seeds agree on a single prediction.
    """
    S = len(per_seed_logits)
    if S == 0:
        raise ValueError("per_seed_logits must contain at least one seed.")
    if S == 1:
        # Single seed — no disagreement possible; return zeros
        N = per_seed_logits[0].shape[0]
        final_pred = _to_f32(per_seed_logits[0]).argmax(axis=-1).astype(np.int32)
        return (
            np.zeros(N, dtype=np.float32),
            final_pred,
            np.ones(N, dtype=bool),
        )

    N = per_seed_logits[0].shape[0]
    # Stack final-epoch predictions from every seed: [S, N]
    final_preds = np.stack(
        [_to_f32(sl).argmax(axis=-1) for sl in per_seed_logits],
        axis=0,
    ).astype(np.int32)                                      # [S, N]

    # Majority vote per sample across seeds
    majority_pred = np.apply_along_axis(
        lambda col: np.bincount(col).argmax(),
        axis=0,
        arr=final_preds,
    ).astype(np.int32)                                      # [N]

    # Fraction of seeds that disagree with the majority
    disagree = (final_preds != majority_pred[None, :])      # [S, N] bool
    ensemble_dis = disagree.mean(axis=0).astype(np.float32)  # [N]

    seed_agree = (ensemble_dis == 0.0)                      # [N]  all S seeds agree

    return ensemble_dis, majority_pred, seed_agree


# ---------------------------------------------------------------------------
# Correctness scalar  (derived from correct_array — companion to ⑤)
# ---------------------------------------------------------------------------

def compute_correctness(correct_array: np.ndarray) -> np.ndarray:
    """
    Fraction of epochs in which the sample was classified correctly.

    correctness[i] = mean(correct_array[i, :])  ∈ [0, 1]
    1.0 = always correct; 0.0 = never correct.

    This is the 'correctness' axis of Dataset Cartography
    (Swayamdipta et al., EMNLP 2020).
    """
    return correct_array.astype(np.float32).mean(axis=1)


# ---------------------------------------------------------------------------
# Main entry point — derive ALL metrics in one call
# ---------------------------------------------------------------------------

def derive_all_metrics(
    logits_array: np.ndarray,
    loss_array: np.ndarray,
    correct_array: np.ndarray,
    grad_norm_array: np.ndarray,
    embeddings_array: np.ndarray,
    labels: np.ndarray,
    *,
    k_mct: int = 5,
    e_early: int = 5,
    knn_k: int = 10,
    # Multi-seed inputs (optional — pass None to skip ensemble metrics)
    per_seed_final_logits: list[np.ndarray] | None = None,
    sample_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Derive all 14 DataGenome metrics from the 5 raw training signals.

    Parameters
    ----------
    logits_array          : [N, T, C]  float16 or float32
    loss_array            : [N, T]     float16 or float32
    correct_array         : [N, T]     uint8
    grad_norm_array       : [N, K]     float16 or float32  (first K epochs)
    embeddings_array      : [N, D]     float16 or float32
    labels                : [N]        int64  — true class index
    k_mct                 : int        stability window for MCT (default 5)
    e_early               : int        early epochs for EL2N (default 5)
    knn_k                 : int        neighbours for kNN density/depth (default 10)
    per_seed_final_logits : list of [N, C] arrays — last-epoch logits from each seed.
                            Pass None (default) when only a single seed is available;
                            ensemble_dis will be set to 0.0 and a warning is printed.
    sample_ids            : [N] array of original sample identifiers for the index.
                            Defaults to 0…N-1 if not provided.

    Returns
    -------
    pd.DataFrame with columns:
        sample_id, label,
        aum, rho, fe, correctness, variability, el2n, grand,
        margin_var, knn_density, prototypicality, ensemble_dis
    Trajectory arrays (margin_traj, confidence_t, loss_traj) and kNN
    intermediates (knn_raw_distances, knn_global_mean) are attached
    via df.attrs for downstream use.
    """
    N, T, C = logits_array.shape
    labels = np.asarray(labels, dtype=np.int64)

    if sample_ids is None:
        sample_ids = np.arange(N)

    # ── Group 1: from logits_array ──────────────────────────────────────────
    margin_traj, margin_var   = compute_margin_trajectory(logits_array, labels)  # ① ⑩
    aum                       = compute_aum(margin_traj)                          # ②
    rho                       = compute_mct_rho(margin_traj, k_mct=k_mct)        # ③
    confidence_t, variability = compute_confidence_trajectories(logits_array, labels)  # ⑤ ⑥
    el2n                      = compute_el2n(logits_array, labels, e_early=e_early)    # ⑦

    # ── Group 2: from correct_array ─────────────────────────────────────────
    fe          = compute_forgetting_events(correct_array)   # ④
    correctness = compute_correctness(correct_array)         # (companion scalar)

    # ── Group 3: from grad_norm_array ───────────────────────────────────────
    grand = compute_grand(grad_norm_array)                   # ⑧

    # ── Group 4: from loss_array (pass-through) ─────────────────────────────
    loss_traj = get_loss_trajectory(loss_array)              # ⑨

    # ── Group 5: from embeddings_array ──────────────────────────────────────
    knn_density, knn_raw_distances, knn_global_mean = compute_knn_density(embeddings_array, k=knn_k)  # ⑪
    prototypicality, centroids   = compute_prototypicality(embeddings_array, labels)             # ⑫

    # ── Group 6: ensemble disagreement (multi-seed) ─────────────────────────
    if per_seed_final_logits is not None:
        ensemble_dis, majority_pred, _ = compute_ensemble_disagreement(
            per_seed_final_logits, labels
        )                                                    # ⑭
    else:
        import warnings
        warnings.warn(
            "per_seed_final_logits not provided — ensemble_dis set to NaN for all samples. "
            "Run at least 3 seeds and pass per_seed_final_logits to enable this metric.",
            UserWarning,
            stacklevel=2,
        )
        ensemble_dis = np.full(N, np.nan, dtype=np.float32)

    # ── Assemble scalar DataFrame ────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "sample_id"      : sample_ids,
            "label"          : labels,
            # ②
            "aum"            : aum.astype(np.float32),
            # ③
            "rho"            : rho,
            # ④
            "fe"             : fe,
            # companion scalar (Dataset Cartography correctness axis)
            "correctness"    : correctness.astype(np.float32),
            # ⑥
            "variability"    : variability.astype(np.float32),
            # ⑦
            "el2n"           : el2n,
            # ⑧
            "grand"          : grand.astype(np.float32),
            # ⑩
            "margin_var"     : margin_var.astype(np.float32),
            # ⑪
            "knn_density"    : knn_density,
            # ⑫
            "prototypicality": prototypicality,
            # ⑭
            "ensemble_dis"   : ensemble_dis,
        }
    )
    df.set_index("sample_id", inplace=True)

    # Attach full trajectories as frame metadata for downstream callers
    # (not stored in parquet — save separately as .npy if persistence is needed)
    df.attrs["margin_traj"]         = margin_traj          # [N, T]  ①
    df.attrs["confidence_t"]        = confidence_t         # [N, T]  ⑤
    df.attrs["loss_traj"]           = loss_traj            # [N, T]  ⑨
    df.attrs["centroids"]           = centroids            # [C, D]  class centroids
    df.attrs["knn_raw_distances"]   = knn_raw_distances    # [N]     raw mean kNN dist
    df.attrs["knn_global_mean"]     = knn_global_mean      # float   normalisation constant

    return df