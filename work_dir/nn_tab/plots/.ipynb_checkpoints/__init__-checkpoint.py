"""Plots module for nntab package.

This module contains visualization functions for model evaluation and training analysis.
"""

from .plots import (
    auc_plot,
    plot_info,
    ttnr_tdr,
    ttnr_tfpr,
    ttnr_fraud_bps,
    tnxs_plots,
    plot_loss_curve
)

__all__ = [
    'auc_plot',
    'plot_info',
    'ttnr_tdr',
    'ttnr_tfpr', 
    'ttnr_fraud_bps',
    'tnxs_plots',
    'plot_loss_curve'
]