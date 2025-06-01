"""
Model implementations for synthetic data generation.

This module contains CTGAN and ensemble model implementations
with privacy-preserving capabilities.
"""

try:
    from .privacy_ctgan import EnsembleCTGAN
except ImportError:
    pass

__all__ = ["EnsembleCTGAN"]
