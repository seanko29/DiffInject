# DiffInject Quick Start Guide

This guide will help you get started with DiffInject in under 30 minutes.


## Step 1: Installation

### Option A: Automated Installation (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd diffinject

# Run the installation script
bash install.sh
```

### Option B: Manual Installation
```bash
# Install main dependencies
pip install -r requirements.txt

# Install classifier training dependencies
cd train_classifier
pip install -r requirements.txt
cd ..

# Install the package
pip install -e .
```

## Step 2: Download Datasets and Models

### Download Datasets
Download one of the supported datasets:

- **BFFHQ**: [Download Link]([https://drive.google.com/file/d/1ZWWjXxcDVK_dATo3zbtHgYgEXn_NVrqm/view?usp=drive_link](https://drive.google.com/file/d/1DuZkhQMstWk0nupeYzX9GeIGTAz7M04g/view?usp=drive_link))
- **CMNIST**: [Download Link](https://drive.google.com/file/d/1ruoc6RC8Lm7QdItAz0pxL_yIkP4G37cV/view?usp=drive_link)
- **DOGSnCATS** [Download Link]((https://drive.google.com/file/d/1DiepWrnFiDn8dzngpL7Mik849tygGlkf/view?usp=drive_link))
### Download Pretrained Models
Download the appropriate pretrained diffusion model for your dataset:

- **FFHQ**: [Download Link](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH)
- **CelebA**: [Download Link](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH)

Place the models in the `pretrained/` directory.

## Step 3: Configure Paths

1. Copy the configuration template:
```bash
cp configs/paths_config_template.py configs/paths_config.py
```

2. Edit `configs/paths_config.py` with your actual paths:
```python
DATASET_PATHS = {
    'bffhq': '/path/to/your/bffhq/dataset',
    # ... other datasets
}

PRETRAINED_MODELS = {
    'ffhq': '/path/to/your/ffhq_model.pt',
    # ... other models
}
```

## Step 4: Prepare Test Images

Create test image directories and add some sample images:

```bash
mkdir -p test_images/bffhq/contents
mkdir -p test_images/bffhq/styles

# Add some images to these directories
# contents/ should contain bias-aligned samples
# styles/ should contain bias-conflict samples
```

## Step 5: Run the Complete Pipeline

### Option A: Automated Pipeline (Recommended)
```bash
# Run the complete pipeline
bash run_pipeline.sh
```

### Option B: Step-by-Step Pipeline

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

## Step 6: View Results

Check the generated results:

```bash
# View synthetic samples
ls bin/bffhq_men/h_gamma_0.7/

# View training results
ls train_classifier/results/bffhq_0.5pct/

# View debiased classifier results
ls train_classifier/results/bffhq_0.5pct_debiased/
```

## Common Issues and Solutions

### Out of Memory Error
```bash
# Reduce batch size in train_classifier.py
# Or use smaller image size
```

### CUDA Not Found
```bash
# Install CUDA toolkit
# Or use CPU-only mode (slower)
```

### Dataset Not Found
```bash
# Check paths in configs/paths_config.py
# Ensure dataset is properly downloaded and extracted
```

## Next Steps

1. **Experiment with different h_gamma values** (0.1, 0.3, 0.5, 0.7, 0.9)
2. **Try different datasets** (CMNIST, CIFAR10C, BAR)
3. **Adjust bias_conflict_ratio** for optimal results
4. **Visualize synthetic samples** to assess quality
5. **Compare bias metrics** before and after debiasing

## Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Review [DOCUMENTATION.md](DOCUMENTATION.md) for technical details
- Look at example scripts in the repository
- Check the `train_classifier/README.md` for dataset-specific information

## Example Output

After running the pipeline, you should see:

```
Results are available in:
  - Biased classifier: ./train_classifier/results/bffhq_0.5pct/
  - Synthetic samples: ./bin/bffhq_men/
  - Debiased classifier: ./train_classifier/results/bffhq_0.5pct_debiased/
```

The synthetic samples will be in directories like `h_gamma_0.1/`, `h_gamma_0.3/`, etc., showing the effect of different content injection strengths. 
