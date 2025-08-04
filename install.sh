#!/bin/bash

# DiffInject Installation Script
# This script sets up the environment for the DiffInject project

set -e  # Exit on any error

echo "=========================================="
echo "DiffInject Installation Script"
echo "=========================================="

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Check if CUDA is available (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    echo "⚠ CUDA not detected. GPU acceleration will not be available."
fi

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv diffinject_env
    source diffinject_env/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install main requirements
echo "Installing main requirements..."
pip install -r requirements.txt

# Install classifier training requirements
echo "Installing classifier training requirements..."
cd train_classifier
pip install -r requirements.txt
cd ..

# Install the package in development mode
echo "Installing DiffInject package..."
pip install -e .

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p pretrained
mkdir -p test_images
mkdir -p results
mkdir -p bin
mkdir -p runs

echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download your datasets and place them in the appropriate directories"
echo "2. Download pretrained diffusion models and place them in ./pretrained/"
echo "3. Update configs/paths_config.py with your dataset and model paths"
echo "4. Run the pipeline:"
echo "   - Train biased classifier: cd train_classifier && python train_classifier.py --dataset <dataset> --pct <percentage>"
echo "   - Extract bias samples: python top_k_loss.py --dataset <dataset> --pct <percentage>"
echo "   - Generate synthetic samples: bash script_diffstyle.sh"
echo "   - Train debiased classifier: python train_classifier.py --dataset <dataset> --pct <percentage> --synthetic_root <path>"
echo ""
echo "For more information, see README.md" 