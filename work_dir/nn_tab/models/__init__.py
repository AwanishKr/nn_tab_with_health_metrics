"""Models module for nntab package.

This module contains neural network architectures and model initialization functions.
"""

from .model import (
    fraudmodel_3layer,
    fraudmodel_5layer,
    fraudmodel_7layer,
    fraudmodel_8layer,
    get_compression_ratio
)
from .model_initialise import get_model

__all__ = [
    'fraudmodel_3layer',
    'fraudmodel_5layer',
    'fraudmodel_7layer',
    'fraudmodel_8layer',
    'get_model',
    'get_compression_ratio'
]