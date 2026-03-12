"""Utils module for nntab package.

This module contains utility functions for training, validation, model management,
score calculation, and curriculum learning utilities.
"""

from .utils import (
    check_for_invalid_values,
    val_fn,
    train_fn,
    train_model,
    train_model_crl
)

from .calculate_scores import (
    add_logits_to_aum_dict,
    calculate_aum,
    compute_grand_score,
    EL2N_score,
    update_forgetting,
    prediction_depth_knn
)

from .crl_utils import (
    negative_entropy,
    ForgettingTracker,
    History
)

__all__ = [
    # Core training utilities
    'check_for_invalid_values',
    'val_fn',
    'train_fn',
    'train_model',
    'train_model_crl',
    
    # Score calculation functions
    'add_logits_to_aum_dict',
    'calculate_aum',
    'compute_grand_score',
    'EL2N_score',
    'update_forgetting',
    'prediction_depth_knn',
    
    # Curriculum learning utilities
    'negative_entropy',
    'ForgettingTracker',
    'History'
]