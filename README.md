# Synthetic GAN - Privacy-Preserving Synthetic Data Generation

**Privacy-Preserving Ensemble CTGAN for Tabular Data**

This solution uses an ensemble of privacy-preserving CTGAN models with differential privacy to generate high-quality synthetic tabular data while maintaining strict privacy constraints.

## 🔧 Technical Approach

### Core Strategy
1. **Privacy-Preserving CTGAN**: Custom CTGAN implementation with differential privacy
2. **Ensemble Learning**: Multiple models for improved robustness and diversity
3. **Smart Preprocessing**: Advanced handling of mixed data types and missing values
4. **Post-Processing**: Ensures data consistency and format compliance

### Key Features
- **Differential Privacy**: Gradient clipping and noise injection during training
- **Early Stopping**: Prevents overfitting to training data
- **Diversity Regularization**: Encourages sample diversity in generated data
- **Robust Preprocessing**: Handles missing values, outliers, and categorical encoding
- **Ensemble Methods**: Combines multiple models for better generalization

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd synthetic-GAN

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Basic Usage

```bash
# Run with default settings
python src/scripts/main.py

# Quick test with reduced parameters
python tests/run_test.py
```

### 3. Configuration

Edit `config/config.py` to customize:
- Model hyperparameters
- Privacy settings
- Ensemble configuration
- Preprocessing options

## 📊 Architecture Overview

```
Input Data (data/raw/)
    ↓
Data Preprocessing (src/synthetic_gan/data/)
├── Missing Value Imputation
├── Outlier Detection & Handling
├── Categorical Encoding
└── Numerical Normalization
    ↓
Ensemble Training (src/synthetic_gan/models/)
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
Output: data/synthetic/synthetic_data.csv
```

## ⚙️ Configuration Options

### Model Configuration
```python
MODEL_CONFIG = {
    'epochs': 300,              # Training epochs
    'batch_size': 500,          # Batch size
    'generator_dim': (256, 256), # Generator architecture
    'discriminator_dim': (256, 256), # Discriminator architecture
    'generator_lr': 2e-4,       # Generator learning rate
    'discriminator_lr': 2e-4,   # Discriminator learning rate
    'pac': 10,                  # PAC size for mode coverage
}
```

### Privacy Configuration
```python
PRIVACY_CONFIG = {
    'differential_privacy': True,  # Enable differential privacy
    'target_epsilon': 8.0,         # Privacy budget
    'target_delta': 1e-5,          # Privacy parameter
    'max_grad_norm': 1.0,          # Gradient clipping threshold
    'noise_multiplier': 1.1,       # Noise scale
}
```

## 🔍 Privacy Mechanisms

### 1. Differential Privacy
- **Gradient Clipping**: Bounds sensitivity of gradients
- **Noise Injection**: Adds calibrated Gaussian noise to gradients
- **Privacy Budget**: Controls privacy-utility tradeoff

### 2. Regularization Techniques
- **Early Stopping**: Prevents overfitting to training data
- **Diversity Loss**: Encourages sample diversity in batches
- **Gradient Penalty**: Stabilizes GAN training

### 3. Ensemble Diversity
- **Data Subsampling**: Each model trains on different subset
- **Hyperparameter Variation**: Slight variations in learning rates
- **Weighted Combination**: Balanced ensemble output

## 📈 Performance Optimization

### Computational Efficiency
- **GPU Acceleration**: CUDA support for faster training
- **Mixed Precision**: Reduces memory usage and training time
- **Batch Processing**: Efficient data loading and processing
- **Early Stopping**: Prevents unnecessary computation

### Memory Management
- **Gradient Checkpointing**: Reduces memory usage during training
- **Batch Size Optimization**: Balances memory and performance
- **Data Loading**: Efficient pandas and torch data handling

## 🎯 Compliance

### Privacy Constraints
- **DCR Share < 52%**: Ensured through differential privacy and regularization
- **NNDR Ratio > 0.5**: Achieved through diversity mechanisms and ensemble approach

### Output Format
- **Exact Structure**: Matches original data dimensions and column names
- **Data Types**: Preserves integer/float/categorical types
- **Missing Values**: Handles and generates appropriate missing patterns

## 🧪 Testing

```bash
python tests/run_test.py --data-path your_data.csv
```

## 📦 Dependencies

Main dependencies:
- `torch>=2.0.0` - Deep learning framework
- `ctgan>=0.7.0` - CTGAN implementation
- `pandas>=1.5.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities
- `opacus>=1.4.0` - Differential privacy

See `requirements.txt` for complete list.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.