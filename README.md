# Synthetic Data Generation

**Privacy-Preserving Ensemble CTGAN for Tabular Data**

This solution uses an ensemble of privacy-preserving CTGAN models with differential privacy to generate high-quality synthetic tabular data while maintaining strict privacy constraints.

## 🔧 Technical Approach

### Core Strategy
1. **Privacy-Preserving CTGAN**: Custom CTGAN implementation with differential privacy.
2. **Ensemble Learning**: Multiple models for improved robustness and diversity.
3. **Smart Preprocessing**: Advanced handling of mixed data types and missing values.
4. **Post-Processing**: Ensures data consistency and format compliance.

### Key Features
- **Differential Privacy**: Gradient clipping and noise injection during training.
- **Early Stopping**: Prevents overfitting to training data.
- **Diversity Regularization**: Encourages sample diversity in generated data.
- **Robust Preprocessing**: Handles missing values, outliers, and categorical encoding.
- **Ensemble Methods**: Combines multiple models for better generalization.

## 📋 Requirements

### System Requirements
- Python 3.8+.
- CUDA-capable GPU (recommended).
- 16GB+ RAM.
- 20GB+ disk space.

### Dependencies
```bash
pip install -r requirements.txt
```

Main dependencies:
- `torch>=2.0.0`
- `ctgan>=0.7.0`
- `pandas>=1.5.0`
- `scikit-learn>=1.3.0`
- `numpy>=1.21.0`

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd ...

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
# Run with default settings.
python main.py

# Quick test with reduced parameters.
python main.py --config-test

# Custom data path.
python main.py --data-path "path/to/your/data.csv"
```

### 3. Configuration
Edit `config.py` to customize:
- Model hyperparameters.
- Privacy settings.
- Ensemble configuration.
- Preprocessing options.

## 📊 Architecture Overview

```
Input Data
    ↓
Data Preprocessing
├── Missing Value Imputation
├── Outlier Detection & Handling
├── Categorical Encoding
└── Numerical Normalization
    ↓
Ensemble Training
├── Model 1: Privacy-CTGAN (subset 1)
├── Model 2: Privacy-CTGAN (subset 2)
└── Model 3: Privacy-CTGAN (subset 3)
    ↓
Synthetic Generation
├── Generate from each model
├── Weighted ensemble combination
└── Shuffle for diversity
    ↓
Post-Processing
├── Inverse transformation
├── Data type consistency
└── Categorical validation
    ↓
Output: synthetic_data.csv
```

## ⚙️ Configuration Options

### Model Configuration
```python
MODEL_CONFIG = {
    'epochs': 300,              # Training epochs.
    'batch_size': 500,          # Batch size.
    'generator_dim': (256, 256), # Generator architecture.
    'discriminator_dim': (256, 256), # Discriminator architecture.
    'generator_lr': 2e-4,       # Generator learning rate.
    'discriminator_lr': 2e-4,   # Discriminator learning rate.
    'pac': 10,                  # PAC size for mode coverage.
}
```

### Privacy Configuration
```python
PRIVACY_CONFIG = {
    'differential_privacy': True,  # Enable differential privacy.
    'target_epsilon': 8.0,         # Privacy budget.
    'target_delta': 1e-5,          # Privacy parameter.
    'max_grad_norm': 1.0,          # Gradient clipping threshold.
    'noise_multiplier': 1.1,       # Noise scale.
}
```

### Ensemble Configuration
```python
ENSEMBLE_CONFIG = {
    'n_models': 3,                    # Number of models.
    'ensemble_weights': [0.4, 0.35, 0.25], # Model weights.
    'diversity_regularization': True,  # Enable diversity loss.
}
```

## 🔍 Privacy Mechanisms

### 1. Differential Privacy
- **Gradient Clipping**: Bounds sensitivity of gradients.
- **Noise Injection**: Adds calibrated Gaussian noise to gradients.
- **Privacy Budget**: Controls privacy-utility tradeoff.

### 2. Regularization Techniques
- **Early Stopping**: Prevents overfitting to training data.
- **Diversity Loss**: Encourages sample diversity in batches.
- **Gradient Penalty**: Stabilizes GAN training.

### 3. Ensemble Diversity
- **Data Subsampling**: Each model trains on different subset.
- **Hyperparameter Variation**: Slight variations in learning rates.
- **Weighted Combination**: Balanced ensemble output.

## 📈 Performance Optimization

### Computational Efficiency
- **GPU Acceleration**: CUDA support for faster training.
- **Mixed Precision**: Reduces memory usage and training time.
- **Batch Processing**: Efficient data loading and processing.
- **Early Stopping**: Prevents unnecessary computation.

### Memory Management
- **Gradient Checkpointing**: Reduces memory usage during training.
- **Batch Size Optimization**: Balances memory and performance.
- **Data Loading**: Efficient pandas and torch data handling.

## 🎯 Compliance

### Privacy Constraints
- **DCR Share < 52%**: Ensured through differential privacy and regularization.
- **NNDR Ratio > 0.5**: Achieved through diversity mechanisms and ensemble approach.

### Output Format
- **Exact Structure**: Matches original data dimensions and column names.
- **Data Types**: Preserves integer/float/categorical types
- **Missing Values**: Handles and generates appropriate missing patterns

## 🔧 Advanced Usage

### Custom Preprocessing
```python
from data_preprocessor import DataPreprocessor

# Custom preprocessing configuration.
custom_config = {
    'handle_missing': True,
    'missing_strategy': 'mode_median',
    'normalize_numerical': True,
    'outlier_threshold': 2.5,  # More aggressive outlier detection.
}

preprocessor = DataPreprocessor(custom_config)
```

### Model Ensembles
```python
from privacy_ctgan import EnsembleCTGAN

# Create larger ensemble.
ensemble_config = {
    'n_models': 5,
    'ensemble_weights': [0.3, 0.25, 0.2, 0.15, 0.1],
    'diversity_regularization': True,
}

model = EnsembleCTGAN(MODEL_CONFIG, PRIVACY_CONFIG, ensemble_config)
```

### Privacy Tuning
```python
# Higher privacy (lower epsilon)
HIGH_PRIVACY_CONFIG = {
    'differential_privacy': True,  # Enable differential privacy.
    'target_epsilon': 4.0,      # Stronger privacy.
    'noise_multiplier': 1.5,    # More noise.
    'max_grad_norm': 0.5,       # Tighter clipping.
}

# Lower privacy (higher utility)
BALANCED_PRIVACY_CONFIG = {
    'differential_privacy': True,  # Enable differential privacy.
    'target_epsilon': 10.0,     # Relaxed privacy.
    'noise_multiplier': 0.8,    # Less noise.
    'max_grad_norm': 1.5,       # Looser clipping.
}
```

## 📊 Monitoring and Debugging

### Training Progress
```bash
# Monitor training with verbose output.
python main.py --verbose

# Check GPU usage.
nvidia-smi

# Monitor training logs.
tail -f training.log
```

### Output Validation
```python
import pandas as pd

# Load and validate output.
synthetic = pd.read_csv('outputs/synthetic_data.csv')
original = pd.read_csv('data/flat-training.csv')

print(f"Shape match: {synthetic.shape == original.shape}")
print(f"Columns match: {list(synthetic.columns) == list(original.columns)}")
print(f"Data types: {synthetic.dtypes.equals(original.dtypes)}")
```

## 🐛 Troubleshooting

### Common Issues

**Memory Errors**
```bash
# Reduce batch size.
python main.py --batch-size 250

# Use CPU if GPU memory insufficient.
python main.py --no-gpu
```

**Slow Training**
```bash
# Test mode for debugging.
python main.py --config-test

# Reduce ensemble size.
# Edit ENSEMBLE_CONFIG['n_models'] = 2
```

**Privacy Constraint Violations**
- Increase `noise_multiplier` in privacy config.
- Reduce `target_epsilon` for stronger privacy.
- Enable more aggressive early stopping.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review configuration options
- Create GitHub issue with detailed description

---