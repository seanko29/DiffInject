---------------- install.sh ----------------------

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

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda detected: $(conda --version)"
    read -p "Do you want to use conda environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating conda environment..."
        conda create -n diffinject python=3.11 -y
        # conda activate 대신 source 사용
        source activate diffinject
        echo "✓ Conda environment created and activated"
    fi
fi

# Check if CUDA is available (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    echo "✓ CUDA detected: Driver version $cuda_version"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    
    # Check CUDA compatibility
    if [[ "$cuda_version" < "11.0" ]]; then
        echo "⚠ Warning: CUDA driver version $cuda_version might be too old for PyTorch 2.0.1"
        echo "  Consider updating your CUDA driver or using CPU-only PyTorch"
    fi
else
    echo "⚠ CUDA not detected. GPU acceleration will not be available."
    echo "  Consider installing CPU-only PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
fi

# Create virtual environment (if not using conda)
if ! command -v conda &> /dev/null || [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3.11 -m venv diffinject_env
        source diffinject_env/bin/activate
        echo "✓ Virtual environment created and activated"
    fi
fi

# Upgrade pip and install setuptools
echo "Upgrading pip and installing setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Install NumPy first to avoid compatibility issues
echo "Installing NumPy first..."
pip install numpy==1.24.4

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install main requirements (excluding PyTorch and NumPy)
echo "Installing main requirements..."
# Create a temporary requirements file without PyTorch and NumPy
grep -v "torch\|numpy" requirements.txt > requirements_temp.txt
pip install -r requirements_temp.txt
rm requirements_temp.txt

# Install classifier training requirements (if directory exists)
if [ -d "train_classifier" ]; then
    echo "Installing classifier training requirements..."
    cd train_classifier
    
                 # Install core scientific packages first
             echo "Installing core scientific packages..."
             pip install scipy==1.11.0
             pip install scikit-learn==1.3.0
             pip install scikit-image==0.21.0
             pip install pandas==2.0.3
             
             # Create a temporary requirements file without problematic packages
             grep -v "opencv-python\|numpy\|scipy\|scikit-learn\|scikit-image\|pandas" requirements.txt > requirements_temp.txt
    pip install -r requirements_temp.txt
    
    # Install opencv-python separately with updated version
    pip install opencv-python==4.11.0.86
    
    rm requirements_temp.txt
    cd ..
else
    echo "⚠ train_classifier directory not found, skipping classifier requirements"
fi

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
