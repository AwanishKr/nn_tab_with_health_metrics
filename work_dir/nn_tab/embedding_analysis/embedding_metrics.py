"""Embedding-based metrics: kNN density and prototypicality.

Both functions accept raw numpy arrays so they stay decoupled from training.
Typical usage:

    embeddings = np.load("raw_signals/embeddings.npy")       # [N, D]
    targets    = np.load("raw_signals/logits_array.npy")      # load targets separately
    # or targets from training_samples.parquet / targets_array

    density = compute_knn_density(embeddings, k=10, output_dir="embedding_analysis_results")
    proto   = compute_prototypicality(embeddings, targets, output_dir="embedding_analysis_results")
"""

import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_knn_density(embeddings, k=10, output_dir="embedding_analysis_results"):
    """Compute per-sample kNN embedding density.

    For each sample, finds the mean Euclidean distance to its k nearest
    neighbours, then normalises by the global mean distance so that
    density > 1 means more isolated than average.

    Args:
        embeddings: np.ndarray of shape [N, D] (float16/32).
        k: number of neighbours (default 10).
        output_dir: root output directory (default "embedding_analysis_results").

    Returns:
        dict with:
            raw_distances:       np.ndarray [N] — unnormalised mean kNN distance per sample.
            normalised_density:  np.ndarray [N] — distance / global_mean_distance.
                                 Values > 1 → more isolated (rare/outlier).
                                 Values < 1 → more crowded (redundant).
            global_mean_distance: float — the normalisation constant.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    N = embeddings.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Column 0 is self-distance (≈0); take columns 1..k
    mean_knn_dist = distances[:, 1:].mean(axis=1)  # [N]

    global_mean = mean_knn_dist.mean()
    # Guard against degenerate case (all identical embeddings)
    if global_mean == 0:
        normalised = np.zeros(N, dtype=np.float32)
    else:
        normalised = mean_knn_dist / global_mean

    # Save results
    save_dir = os.path.join(output_dir, "knn_density")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "raw_distances.npy"), mean_knn_dist)
    np.save(os.path.join(save_dir, "normalised_density.npy"), normalised)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({"k": k, "global_mean_distance": float(global_mean), "num_samples": N}, f, indent=2)

    return {
        "raw_distances": mean_knn_dist,
        "normalised_density": normalised,
        "global_mean_distance": float(global_mean),
    }


def compute_prototypicality(embeddings, targets, output_dir="embedding_analysis_results"):
    """Compute per-sample prototypicality via cosine similarity to class centroid.

    For each sample, computes the cosine similarity between its embedding and
    the mean embedding (centroid) of its class. High value = prototypical
    member of the class; low value = atypical.

    Args:
        embeddings: np.ndarray of shape [N, D].
        targets:    np.ndarray of shape [N] — integer class labels.
        output_dir: root output directory (default "embedding_analysis_results").

    Returns:
        dict with:
            prototypicality: np.ndarray [N] — cosine similarity to own class centroid.
            class_centroids: dict mapping class_id → np.ndarray [D].
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets).ravel()
    N, D = embeddings.shape

    classes = np.unique(targets)
    centroids = {}
    for c in classes:
        mask = targets == c
        centroids[int(c)] = embeddings[mask].mean(axis=0)

    proto_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        sample = embeddings[i]
        centroid = centroids[int(targets[i])]
        denom = np.linalg.norm(sample) * np.linalg.norm(centroid)
        if denom == 0:
            proto_scores[i] = 0.0
        else:
            proto_scores[i] = np.dot(sample, centroid) / denom

    # Save results
    save_dir = os.path.join(output_dir, "prototypicality")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "prototypicality_scores.npy"), proto_scores)
    # Save centroids as a single .npz with class ids as keys
    np.savez(os.path.join(save_dir, "class_centroids.npz"),
             **{str(c): v for c, v in centroids.items()})

    return {
        "prototypicality": proto_scores,
        "class_centroids": centroids,
    }
