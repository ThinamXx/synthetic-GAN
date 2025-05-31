#!/usr/bin/env python3

"""
Main script for synthetic data generation.
Privacy-preserving synthetic data generation using ensemble CTGAN.
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import our custom modules.
from config import *
from data_preprocessor import DataPreprocessor
from privacy_ctgan import EnsembleCTGAN


def load_data(filepath):
    """Load training data"""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Data shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    return data


def validate_privacy_constraints(synthetic_data, original_data):
    """
    Placeholder for privacy validation.
    """
    print("Privacy constraint validation (placeholder):")
    print("- DCR Share: < 52% (to be evaluated by competition system)")
    print("- NNDR Ratio: > 0.5 (to be evaluated by competition system)")
    return True


def post_process_synthetic_data(synthetic_data, preprocessor):
    """Post-process synthetic data to ensure consistency."""
    print("Post-processing synthetic data...")

    # Inverse transform to original format.
    processed_data = preprocessor.inverse_transform(synthetic_data)

    # Ensure integer columns are integers.
    for col in preprocessor.numerical_columns:
        if col in processed_data.columns:
            # Check if original data was integer.
            original_col_data = processed_data[col]
            if original_col_data.dtype in ["int64", "int32"]:
                processed_data[col] = processed_data[col].round().astype("int64")

    # Ensure categorical columns have valid values.
    for col in preprocessor.categorical_columns:
        if col in processed_data.columns:
            # Ensure no invalid categorical values.
            valid_categories = preprocessor.column_info[col]["unique_values"]
            mask = processed_data[col].isin(valid_categories)
            if not mask.all():
                # Replace invalid values with most frequent.
                most_frequent = (
                    processed_data[col].mode().iloc[0]
                    if len(processed_data[col].mode()) > 0
                    else valid_categories[0]
                )
                processed_data.loc[~mask, col] = most_frequent

    return processed_data


def generate_synthetic_data():
    """Main function to generate synthetic data"""
    start_time = time.time()

    print("=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("Privacy-Preserving Ensemble CTGAN")
    print("=" * 60)

    # Create output directory.
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load data.
    data = load_data(DATA_PATH)

    # Initialize preprocessor.
    print("\n" + "=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
    processed_data = preprocessor.fit_transform(data)

    # Save preprocessor for reproducibility.
    preprocessor.save_preprocessor(output_dir / "preprocessor.pkl")

    print(f"Processed data shape: {processed_data.shape}")

    # Split for validation (simulate train/validation split).
    validation_split = TRAINING_CONFIG["validation_split"]
    n_val = int(len(processed_data) * validation_split)
    n_train = len(processed_data) - n_val

    indices = np.random.permutation(len(processed_data))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = processed_data.iloc[train_indices].reset_index(drop=True)
    val_data = processed_data.iloc[val_indices].reset_index(drop=True)

    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # Initialize ensemble model.
    print("\n" + "=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)

    ensemble_model = EnsembleCTGAN(MODEL_CONFIG, PRIVACY_CONFIG, ENSEMBLE_CONFIG)

    # Train the ensemble.
    ensemble_model.fit(train_data)

    # Generate synthetic data.
    print("\n" + "=" * 50)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 50)

    print(f"Generating {SAMPLE_SIZE} synthetic samples...")
    synthetic_data = ensemble_model.sample(SAMPLE_SIZE)

    # Post-process synthetic data.
    synthetic_data = post_process_synthetic_data(synthetic_data, preprocessor)

    # Validate structure matches original.
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Original columns: {len(data.columns)}")
    print(f"Synthetic columns: {len(synthetic_data.columns)}")

    assert synthetic_data.shape[1] == data.shape[1], "Column count mismatch!"
    assert list(synthetic_data.columns) == list(data.columns), "Column names mismatch!"

    # Save synthetic data.
    output_path = output_dir / OUTPUT_PATH
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")

    # Privacy validation (placeholder).
    print("\n" + "=" * 50)
    print("PRIVACY VALIDATION")
    print("=" * 50)

    validate_privacy_constraints(synthetic_data, data)

    # Performance summary.
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY")
    print("=" * 50)
    print(
        f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)"
    )
    print(f"Generated samples: {len(synthetic_data)}")
    print(f"Original data size: {len(data)}")
    print(f"Privacy constraints: DCR Share < 52%, NNDR Ratio > 0.5")
    print(f"Model type: Privacy-preserving Ensemble CTGAN")
    print(f"Number of ensemble models: {ENSEMBLE_CONFIG['n_models']}")

    if total_time > COMPUTE_CONFIG["max_runtime_hours"] * 3600:
        print("WARNING: Execution time exceeded target runtime!")
    else:
        print("✓ Execution completed within time constraints")

    print("\n" + "=" * 60)
    print("COMPETITION SUBMISSION READY!")
    print(f"Submit file: {output_path}")
    print("=" * 60)

    return synthetic_data


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for competition"
    )
    parser.add_argument(
        "--config-test",
        action="store_true",
        help="Run quick test with reduced parameters",
    )
    parser.add_argument(
        "--data-path", type=str, default=DATA_PATH, help="Path to training data"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=OUTPUT_PATH,
        help="Output filename for synthetic data",
    )

    args = parser.parse_args()

    # Override config for testing.
    if args.config_test:
        print("Running in TEST mode with reduced parameters...")
        MODEL_CONFIG["epochs"] = 50
        MODEL_CONFIG["batch_size"] = 200
        ENSEMBLE_CONFIG["n_models"] = 2
        SAMPLE_SIZE = 10000

    # Override paths if provided.
    if args.data_path != DATA_PATH:
        globals()["DATA_PATH"] = args.data_path
    if args.output_path != OUTPUT_PATH:
        globals()["OUTPUT_PATH"] = args.output_path

    # Set random seeds for reproducibility.
    np.random.seed(42)

    try:
        synthetic_data = generate_synthetic_data()
        print("\n✓ SUCCESS: Synthetic data generation completed!")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
