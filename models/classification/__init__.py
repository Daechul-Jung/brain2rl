"""
Classification models for brain signals
"""

from .action_classifier import (
    EEGConvNet,
    FMRIConvNet,
    train_classifier,
    evaluate_classifier
) 