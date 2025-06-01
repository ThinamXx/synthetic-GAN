"""
Configuration file for the Synthetic Data Generation.
"""

# Data configuration.
DATA_PATH = "data/raw/flat-training.csv"
OUTPUT_PATH = "synthetic_data.csv"
SAMPLE_SIZE = 100000  # Number of synthetic samples to generate.

# Model configuration.
MODEL_CONFIG = {
    # CTGAN parameters optimized for privacy and quality.
    "epochs": 300,
    "batch_size": 500,
    "generator_dim": (256, 256),
    "discriminator_dim": (256, 256),
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
    "discriminator_steps": 1,
    "log_frequency": True,
    "verbose": True,
    "pac": 10,  # Pac size for better mode coverage.
}

# Privacy configuration.
PRIVACY_CONFIG = {
    "differential_privacy": True,
    "target_epsilon": 8.0,  # Privacy budget.
    "target_delta": 1e-5,
    "max_grad_norm": 1.0,  # Gradient clipping.
    "noise_multiplier": 1.1,
}

# Ensemble configuration.
ENSEMBLE_CONFIG = {
    "n_models": 3,  # Number of models in ensemble.
    "ensemble_weights": [0.4, 0.35, 0.25],  # Weights for ensemble.
    "diversity_regularization": True,
}

# Training configuration.
TRAINING_CONFIG = {
    "validation_split": 0.15,
    "early_stopping_patience": 50,
    "min_delta": 0.001,
    "save_checkpoints": True,
    "checkpoint_frequency": 50,
}

# Data preprocessing.
PREPROCESSING_CONFIG = {
    "handle_missing": True,
    "missing_strategy": "mode_median",  # Mode for categorical, median for numerical.
    "normalize_numerical": True,
    "encode_categorical": True,
    "outlier_detection": True,
    "outlier_threshold": 3.0,  # Z-score threshold.
}

# Evaluation thresholds.
EVALUATION_THRESHOLDS = {
    "dcr_share_max": 0.52,  # Must be below 52%.
    "nndr_ratio_min": 0.5,  # Must be above 0.5.
}

# Computational settings.
COMPUTE_CONFIG = {
    "use_gpu": True,
    "mixed_precision": True,
    "num_workers": 4,
    "max_runtime_hours": 5.5,  # Safety margin under 6 hours.
}
