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
    TrainingSignalCollector,
    compute_gradient_norms_pass,
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
    
    # Raw signal collection (DataGenome)
    'TrainingSignalCollector',
    'compute_gradient_norms_pass',
    
    # Curriculum learning utilities
    'negative_entropy',
    'ForgettingTracker',
    'History'
]