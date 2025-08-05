
# Revisiting Debiasing Classifiers via Synthetic Data Generation using Diffusion-based Style Injection

This repository contains the official implementation of **Revisiting Debiasing Classifiers via Synthetic Data Generation using Diffusion-based Style Injection**, a method for mitigating dataset bias through training-free content injection using diffusion models.


## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0.1+
- CUDA 11.7+ (for GPU acceleration)

### Setup
```bash
# Install main dependencies
pip install -r requirements.txt

# Install classifier training dependencies
cd train_classifier
pip install -r requirements.txt
```


## Overview

DiffInject addresses dataset bias by:
1. **Training a biased classifier** on the original dataset to identify bias patterns
2. **Extracting bias-conflict samples** (samples with high loss values) 
3. **Generating synthetic samples** using content injection in diffusion model h-space
4. **Retraining a debiased classifier** on the combined original + synthetic dataset

The method leverages the h-space (bottleneck features) of diffusion models to inject content from bias-conflict samples into bias-aligned samples, creating diverse synthetic data that helps balance the dataset.

## Pipeline

### Step 1: Train Biased Classifier
```bash
cd train_classifier
python train_classifier.py --dataset <dataset_name> --pct <percentage>
```

This trains a classifier on the original dataset, intentionally learning the bias patterns present in the data.

### Step 2: Extract Bias-Conflict Samples
```bash
cd train_classifier
python top_k_loss.py --dataset <dataset_name> --pct <percentage>
```

This extracts the top-K samples with highest loss values, which represent bias-conflict samples where the model struggles to classify correctly.

### Step 3: Generate Synthetic Samples
```bash
# For different h_gamma values (content injection strength)
bash script_diffstyle.sh
# or
bash script_bffhq_men.sh
```

This runs the content injection process using diffusion models to generate synthetic samples by injecting content from bias-conflict samples into bias-aligned samples.

### Step 4: Train Debiased Classifier
```bash
cd train_classifier
python train_classifier.py --dataset <dataset_name> --pct <percentage> --synthetic_root <path_to_synthetic_samples>
```

This retrains the classifier on the combined original dataset and synthetic samples to create a debiased model.


## Datasets

The method supports multiple datasets:
- **CMNIST**: Colored MNIST with bias
- **CIFAR10C**: CIFAR-10 with color bias
- **BFFHQ**: Biased FFHQ dataset
- **BAR**: Biased Attribute Recognition dataset

Download datasets from the provided links in `train_classifier/README.md`.

## Configuration

### Main Diffusion Configuration
Edit `configs/paths_config.py` to set dataset and model paths:
```python
# Example configuration
DATASET_PATHS = {
    'ffhq': '/path/to/ffhq/dataset',
    'celeba': '/path/to/celeba/dataset',
    # ... other datasets
}

PRETRAINED_MODELS = {
    'ffhq': '/path/to/ffhq_model.pt',
    'celeba': '/path/to/celeba_model.pt',
    # ... other models
}
```

### Script Parameters
Key parameters in the generation scripts:
- `h_gamma`: Content injection strength (0.1-0.9)
- `dt_lambda`: Domain transfer parameter
- `t_boost`: Noise injection parameter
- `n_gen_step`: Number of generation steps
- `n_inv_step`: Number of inversion steps

## Usage Examples

### Complete Pipeline for BFFHQ Dataset

1. **Train biased classifier:**
```bash
cd train_classifier
python train_classifier.py --dataset bffhq --pct 0.5pct
```

2. **Extract bias-conflict samples:**
```bash
python top_k_loss.py --dataset bffhq --pct 0.5pct
```

3. **Generate synthetic samples:**
```bash
cd ..
bash script_bffhq_men.sh
```

4. **Train debiased classifier:**
```bash
cd train_classifier
python train_classifier.py --dataset bffhq --pct 0.5pct \
    --synthetic_root ../bin/bffhq_men/h_gamma_0.7 \
    --bias_conflict_ratio 0.1
```

### Custom Dataset

To use with your own dataset:

1. Place your dataset in the appropriate directory structure
2. Update `configs/paths_config.py` with your dataset paths
3. Modify the script parameters in `script_diffstyle.sh` or create a new script
4. Follow the same pipeline steps

## Key Files

- `main.py`: Main diffusion model execution script
- `diffusion_latent.py`: Core diffusion model implementation with h-space manipulation
- `train_classifier/train_classifier.py`: Classifier training implementation
- `train_classifier/top_k_loss.py`: Bias-conflict sample extraction
- `script_*.sh`: Generation scripts for different datasets
- `configs/`: Configuration files for different datasets

## Results

The method generates synthetic samples that help balance the dataset bias. Results are saved in:
- `bin/`: Generated synthetic samples
- `train_classifier/results/`: Training results and model checkpoints
- `runs/`: TensorBoard logs


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation builds upon previous work on diffusion models and dataset bias mitigation. We thank the authors of the referenced papers for their contributions to the field.
