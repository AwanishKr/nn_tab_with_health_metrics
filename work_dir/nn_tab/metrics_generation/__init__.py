"""Post-training metrics computation for DataGenome sample categorisation.

This submodule operates on raw signals collected after training completes.
It has no involvement during training — it consumes the saved .npy arrays
produced by TrainingSignalCollector.

Modules:
    derived_metrics  — DataGenome metrics (margin, AUM, MCT, etc.)
    categorization   — maps metrics to 8 category boolean columns
"""

from .categorization import categorize, save_categorized_report
from .derived_metrics import (
    derive_all_metrics,
    compute_margin_trajectory,
    compute_aum,
    compute_mct_rho,
    compute_forgetting_events,
    compute_confidence_trajectories,
    compute_el2n,
    compute_grand,
    get_loss_trajectory,
    compute_knn_density,
    compute_prototypicality,
    compute_ensemble_disagreement,
    compute_correctness,
)
