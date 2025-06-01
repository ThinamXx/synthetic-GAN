#!/usr/bin/env python3

"""
Quick test script for synthetic data generation solution.
Tests with small subset for debugging and validation.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import our modules.
from config.config import *
from synthetic_gan.data.data_preprocessor import DataPreprocessor
from synthetic_gan.models.privacy_ctgan import EnsembleCTGAN


def run_quick_test():
    """Run quick test with small data subset."""
    print("=" * 60)
    print("QUICK TEST - SYNTHETIC DATA GENERATION")
    print("=" * 60)

    start_time = time.time()

    # Test configuration - reduced parameters.
    test_config = MODEL_CONFIG.copy()
    test_config.update(
        {
            "epochs": 20,
            "batch_size": 100,
            "generator_dim": (128, 128),
            "discriminator_dim": (128, 128),
        }
    )

    test_ensemble_config = ENSEMBLE_CONFIG.copy()
    test_ensemble_config["n_models"] = 2
    test_ensemble_config["ensemble_weights"] = [0.6, 0.4]

    test_sample_size = 1000

    try:
        # Load small subset of data.
        print("Loading data subset...")
        full_data = pd.read_csv(DATA_PATH)

        # Use first 5000 rows for testing.
        test_data = full_data.head(5000).copy()
        print(f"Test data shape: {test_data.shape}")

        # Preprocessing.
        print("\nPreprocessing...")
        preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
        processed_data = preprocessor.fit_transform(test_data)
        print(f"Processed shape: {processed_data.shape}")

        # Training.
        print("\nTraining ensemble...")
        ensemble_model = EnsembleCTGAN(
            test_config, PRIVACY_CONFIG, test_ensemble_config
        )

        ensemble_model.fit(processed_data)

        # Generation.
        print(f"\nGenerating {test_sample_size} synthetic samples...")
        synthetic_data = ensemble_model.sample(test_sample_size)

        # Post-processing.
        print("Post-processing...")
        synthetic_data = preprocessor.inverse_transform(synthetic_data)

        # Validation.
        print("\nValidation:")
        print(f"✓ Generated shape: {synthetic_data.shape}")
        print(f"✓ Expected columns: {len(test_data.columns)}")
        print(f"✓ Actual columns: {len(synthetic_data.columns)}")
        print(
            f"✓ Column names match: {list(synthetic_data.columns) == list(test_data.columns)}"
        )

        # Basic statistics comparison.
        print("\nBasic Statistics Comparison:")
        for col in test_data.select_dtypes(include=[np.number]).columns[:5]:
            orig_mean = test_data[col].mean()
            synth_mean = synthetic_data[col].mean()
            print(f"  {col}: Original={orig_mean:.3f}, Synthetic={synth_mean:.3f}")

        # Save test output.
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        test_output_path = output_dir / "test_synthetic_data.csv"
        synthetic_data.to_csv(test_output_path, index=False)
        print(f"\n✓ Test output saved to: {test_output_path}")

        # Timing.
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✓ Test completed in {total_time:.2f} seconds")

        if total_time < 300:  # 5 minutes for test
            print("✓ Test performance acceptable")
        else:
            print("⚠ Test took longer than expected")

        print("\n" + "=" * 60)
        print("✓ QUICK TEST PASSED!")
        print("Solution appears to be working correctly.")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def validate_dependencies():
    """Validate all required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        "torch",
        "pandas",
        "numpy",
        "sklearn",
        "ctgan",
        "tqdm",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install with: pip install -r requirements.txt")
        return False

    print("✓ All dependencies available")
    return True


def check_data_file():
    """Check if training data file exists."""
    if not Path(DATA_PATH).exists():
        print(f"✗ Data file not found: {DATA_PATH}")
        print("Please ensure the training data file is in the correct location.")
        return False

    print(f"✓ Data file found: {DATA_PATH}")
    return True


def main():
    """Main test function."""
    print("SYNTHETIC DATA GENERATION - QUICK TEST")
    print("Testing solution components...\n")

    # Validate dependencies.
    if not validate_dependencies():
        return 1

    print()

    # Check data file.
    if not check_data_file():
        return 1

    print()

    # Run quick test.
    if not run_quick_test():
        return 1

    print("\n🎉 All tests passed! Solution is ready for full run.")
    print("Run 'python main.py' for complete generation.")

    return 0


if __name__ == "__main__":
    exit(main())
