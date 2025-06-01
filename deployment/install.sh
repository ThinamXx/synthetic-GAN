#!/bin/bash

echo "=============================================="
echo "Synthetic Data Generation Setup"
echo "=============================================="

# Check Python version.
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.8+ is available.
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.8+ is required"
    exit 1
fi

# Check if pip is available.
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is available"
else
    echo "✗ pip3 is not available. Please install pip."
    exit 1
fi

# Create virtual environment.
echo ""
echo "Creating virtual environment..."
if python3 -m venv synthetic_env; then
    echo "✓ Virtual environment created"
    echo "To activate: source synthetic_env/bin/activate"
else
    echo "⚠ Virtual environment creation failed, continuing with system Python"
fi

# Install dependencies.
echo ""
echo "Installing dependencies..."
if pip3 install -r requirements.txt; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Check CUDA availability.
echo ""
echo "Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  PyTorch version: {torch.__version__}')
else:
    print('⚠ CUDA not available, will use CPU')
    print(f'  PyTorch version: {torch.__version__}')
"

# Create output directories.
echo ""
echo "Creating output directories..."
mkdir -p outputs
mkdir -p test_outputs
echo "✓ Output directories created"

# Make scripts executable.
echo ""
echo "Making scripts executable..."
chmod +x main.py
chmod +x run_test.py
echo "✓ Scripts are now executable"

# Check data file.
echo ""
echo "Checking for training data..."
if [ -f "data/flat-training.csv" ]; then
    echo "✓ Training data found"
    file_size=$(ls -lh data/flat-training.csv | awk '{print $5}')
    echo "  File size: $file_size"
else
    echo "⚠ Training data not found at data/flat-training.csv"
    echo "  Please ensure the training data is in the correct location"
fi

echo ""
echo "=============================================="
echo "Installation completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Ensure training data is at: data/flat-training.csv"
echo "2. Run quick test: python3 run_test.py"
echo "3. Run full generation: python3 main.py"
echo ""
echo "For testing with reduced parameters:"
echo "  python3 main.py --config-test"
echo ""
echo "Documentation: See README.md for detailed usage"
echo "==============================================" 