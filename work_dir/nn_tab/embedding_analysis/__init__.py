"""Post-training embedding analysis for DataGenome sample categorisation.

This submodule operates on embeddings collected after training completes.
It has no involvement during training — it consumes the saved .npy arrays
produced by TrainingSignalCollector.

Modules:
    embedding_metrics  — kNN density and prototypicality scores
    prediction_depth   — prediction depth probing (TBD)
"""

from .embedding_metrics import compute_knn_density, compute_prototypicality
