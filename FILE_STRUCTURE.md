# DiffInject File Structure

This document provides an overview of the DiffInject codebase organization.

## Root Directory

```
diffinject/
├── README.md                    # Main documentation and overview
├── QUICKSTART.md               # Quick start guide for new users
├── DOCUMENTATION.md            # Technical documentation
├── FILE_STRUCTURE.md           # This file - codebase organization
├── setup.py                    # Package installation configuration
├── requirements.txt            # Main Python dependencies
├── install.sh                  # Automated installation script
├── run_pipeline.sh             # Complete pipeline example script
├── .gitignore                  # Git ignore rules
├── main.py                     # Main diffusion model execution script
├── diffusion_latent.py         # Core diffusion model with h-space manipulation
├── script_diffstyle.sh         # Sample generation script for general use
├── script_bffhq_men.sh         # Sample generation script for BFFHQ dataset
└── tmp_script.sh               # Temporary script template
```

## Core Directories

### `configs/` - Configuration Files
```
configs/
├── paths_config.py             # Dataset and model path configurations
├── paths_config_template.py    # Template for path configuration
├── ffhq.yml                    # FFHQ dataset configuration
├── celeba.yml                  # CelebA dataset configuration
├── afhq.yml                    # AFHQ dataset configuration
├── metface.yml                 # MetFaces dataset configuration
├── imagenet.yml                # ImageNet dataset configuration
├── custom.yml                  # Custom dataset configuration
├── church.yml                  # LSUN Church dataset configuration
├── bedroom.yml                 # LSUN Bedroom dataset configuration
└── celeba_dialog.yml           # CelebA dialog configuration
```

### `train_classifier/` - Classifier Training Module
```
train_classifier/
├── README.md                   # Classifier training documentation
├── requirements.txt            # Classifier-specific dependencies
├── train_classifier.py         # Main classifier training script
├── top_k_loss.py              # Bias-conflict sample extraction
├── models.py                   # Model architectures (MLP, ResNet18)
├── utils.py                    # Utility functions
├── aug_diffstyle.py           # Data augmentation for diffusion style
└── results/                    # Training results and checkpoints
```

### `models/` - Model Definitions
```
models/
├── __init__.py
├── diffusion.py               # Diffusion model implementations
├── unet.py                    # U-Net architecture
└── classifier.py              # Classifier model definitions
```

### `datasets/` - Dataset Handling
```
datasets/
├── __init__.py
├── base.py                    # Base dataset class
├── ffhq.py                    # FFHQ dataset implementation
├── celeba.py                  # CelebA dataset implementation
└── custom.py                  # Custom dataset implementation
```

### `losses/` - Loss Functions
```
losses/
├── __init__.py
├── diffusion_loss.py          # Diffusion model losses
├── classifier_loss.py         # Classifier training losses
└── bias_loss.py               # Bias-aware loss functions
```

### `utils/` - Utility Functions
```
utils/
├── __init__.py
├── image_utils.py             # Image processing utilities
├── metrics.py                 # Evaluation metrics
├── visualization.py           # Visualization tools
└── config_utils.py            # Configuration utilities
```

### `runs/` - Experiment Tracking
```
runs/                          # TensorBoard logs and experiment tracking
```

### `bin/` - Generated Outputs
```
bin/                           # Generated synthetic samples
├── h_gamma_0.1/              # Samples with h_gamma=0.1
├── h_gamma_0.3/              # Samples with h_gamma=0.3
├── h_gamma_0.5/              # Samples with h_gamma=0.5
├── h_gamma_0.7/              # Samples with h_gamma=0.7
└── h_gamma_0.9/              # Samples with h_gamma=0.9
```

### `test_images/` - Test Images
```
test_images/                   # Test images for content injection
├── bffhq/
│   ├── contents/             # Bias-aligned samples
│   └── styles/               # Bias-conflict samples
├── celeba/
│   ├── contents/
│   └── styles/
└── afhq/
    ├── contents/
    └── styles/
```

### `src/` - Source Assets
```
src/
└── teaser.png                # Teaser image for documentation
```

## Key Files Explained

### Core Implementation Files

1. **`main.py`**: Entry point for the diffusion model pipeline
   - Handles command-line arguments
   - Orchestrates the content injection process
   - Manages different execution modes

2. **`diffusion_latent.py`**: Core diffusion model implementation
   - Implements DDIM with h-space manipulation
   - Handles content injection in bottleneck features
   - Manages the generation process

3. **`train_classifier/train_classifier.py`**: Classifier training
   - Trains biased classifiers on original datasets
   - Supports multiple model architectures
   - Implements bias-aware training strategies

4. **`train_classifier/top_k_loss.py`**: Bias sample extraction
   - Identifies bias-conflict samples
   - Extracts samples with highest loss values
   - Prepares data for synthetic generation

### Configuration Files

1. **`configs/paths_config.py`**: Path configuration
   - Dataset paths
   - Pretrained model paths
   - Output directory paths

2. **`configs/*.yml`**: Dataset-specific configurations
   - Model parameters
   - Diffusion parameters
   - Training hyperparameters

### Script Files

1. **`script_diffstyle.sh`**: General sample generation
   - Runs content injection for multiple h_gamma values
   - Configurable for different datasets

2. **`script_bffhq_men.sh`**: BFFHQ-specific generation
   - Optimized for BFFHQ dataset
   - Pre-configured parameters

3. **`run_pipeline.sh`**: Complete pipeline example
   - Demonstrates full DiffInject workflow
   - From biased training to debiased classifier

### Documentation Files

1. **`README.md`**: Main documentation
   - Overview of the method
   - Installation instructions
   - Usage examples

2. **`QUICKSTART.md`**: Quick start guide
   - Step-by-step setup
   - Common issues and solutions

3. **`DOCUMENTATION.md`**: Technical details
   - Implementation architecture
   - Hyperparameter tuning
   - Troubleshooting guide

## File Naming Conventions

- **Python files**: snake_case (e.g., `train_classifier.py`)
- **Configuration files**: lowercase with extensions (e.g., `ffhq.yml`)
- **Script files**: descriptive names with `.sh` extension
- **Directories**: lowercase with underscores for spaces

## Dependencies

### Main Dependencies (`requirements.txt`)
- PyTorch 2.0.1+
- NumPy, Pillow, OpenCV
- PyYAML, tqdm
- CUDA support (optional)

### Classifier Dependencies (`train_classifier/requirements.txt`)
- Additional ML libraries
- Evaluation metrics
- Visualization tools

## Output Structure

After running the pipeline, the following structure is created:

```
results/
├── biased_classifier/         # Original biased classifier results
├── synthetic_samples/         # Generated synthetic samples
└── debiased_classifier/       # Final debiased classifier results

bin/
├── h_gamma_0.1/              # Synthetic samples with different injection strengths
├── h_gamma_0.3/
├── h_gamma_0.5/
├── h_gamma_0.7/
└── h_gamma_0.9/

runs/
└── tensorboard_logs/          # Training logs and metrics
```

This structure ensures organized outputs and easy comparison between different experimental settings. 