"""
Synthetic GAN - Privacy-Preserving Synthetic Data Generation

A comprehensive framework for generating synthetic tabular data using
privacy-preserving CTGAN models with differential privacy.
"""

__version__ = "1.0.0"
__author__ = "Thinam Tamang"
__email__ = ""

# Import core components for easy access.
try:
    from .models.privacy_ctgan import EnsembleCTGAN
    from .data.data_preprocessor import DataPreprocessor
except ImportError:
    # Handle import errors gracefully during development
    pass

__all__ = [
    "EnsembleCTGAN",
    "DataPreprocessor",
]
