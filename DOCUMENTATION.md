# DiffInject Technical Documentation

## Overview

DiffInject is a training-free method for mitigating dataset bias using content injection in diffusion model h-space. The method consists of four main components:

1. **Biased Classifier Training**: Identifies bias patterns in the dataset
2. **Bias-Conflict Sample Extraction**: Finds samples that conflict with the learned bias
3. **Content Injection**: Generates synthetic samples using diffusion model h-space manipulation
4. **Debiased Classifier Training**: Retrains on the balanced dataset

## Technical Details

### H-Space Content Injection

The core innovation of DiffInject lies in manipulating the h-space (bottleneck features) of diffusion models. The h-space contains semantic information about the image and can be modified to inject content from one image into another.

#### Key Parameters

- **h_gamma**: Controls the strength of content injection (0.0-1.0)
  - 0.0: No injection (original image)
  - 1.0: Full injection (complete content transfer)
  - 0.5: Balanced injection (recommended starting point)

- **dt_lambda**: Domain transfer parameter for out-of-domain style transfer
- **t_boost**: Noise injection parameter for controlling generation quality
- **n_gen_step**: Number of diffusion generation steps
- **n_inv_step**: Number of inversion steps for finding latent representations

### Implementation Architecture

#### 1. Diffusion Model Integration (`diffusion_latent.py`)

The main diffusion model implementation extends the DDIM (Denoising Diffusion Implicit Models) framework with h-space manipulation capabilities:

```python
class Asyrp:
    def __init__(self, config):
        # Initialize diffusion model
        # Load pretrained weights
        # Setup h-space manipulation
        
    def inject_content(self, content_image, style_image, h_gamma):
        # Extract h-space features from content image
        # Blend with style image h-space features
        # Generate synthetic image
```

#### 2. Classifier Training (`train_classifier/`)

The classifier training module supports multiple architectures:

- **CMNIST**: 3-layer MLP
- **Other datasets**: ResNet18 with modifications

Key features:
- Generalized Cross Entropy (GCE) loss for bias amplification
- Top-K loss extraction for bias-conflict sample identification
- Stratified sampling for balanced training

#### 3. Sample Generation Pipeline

The generation process follows these steps:

1. **Inversion**: Convert input images to latent representations
2. **H-space extraction**: Extract bottleneck features from both content and style images
3. **Content injection**: Blend h-space features using spherical linear interpolation (SLERP)
4. **Generation**: Run the diffusion process with modified h-space features
5. **Post-processing**: Apply any additional modifications (masks, etc.)

### Dataset Support

DiffInject supports multiple datasets with different bias types:

#### CMNIST (Colored MNIST)
- **Bias**: Color-digit correlation
- **Task**: Digit classification
- **Model**: 3-layer MLP

#### CIFAR10C (CIFAR-10 with Color Bias)
- **Bias**: Color-class correlation
- **Task**: Image classification
- **Model**: ResNet18

#### BFFHQ (Biased FFHQ)
- **Bias**: Demographic attributes
- **Task**: Attribute classification
- **Model**: ResNet18

#### BAR (Biased Attribute Recognition)
- **Bias**: Attribute correlations
- **Task**: Multi-attribute classification
- **Model**: ResNet18

### Configuration System

The configuration system uses YAML files for dataset-specific settings:

```yaml
# Example: ffhq.yml
model:
  name: "ffhq"
  image_size: 256
  num_channels: 3

diffusion:
  beta_start: 0.0001
  beta_end: 0.02
  num_timesteps: 1000

injection:
  h_gamma_range: [0.1, 0.3, 0.5, 0.7, 0.9]
  dt_lambda: 0.9985
  t_boost: 200
```

### Performance Optimization

#### GPU Memory Management
- Gradient checkpointing for large models
- Mixed precision training (FP16)
- Dynamic batch sizing based on available memory

#### Computational Efficiency
- Cached latent representations
- Parallel processing for multiple h_gamma values
- Early stopping for convergence detection

### Evaluation Metrics

The method evaluates debiasing effectiveness using:

1. **Accuracy**: Overall classification performance
2. **Bias Metrics**: Demographic parity, equalized odds
3. **Fairness Metrics**: Statistical parity difference, equal opportunity difference
4. **Quality Metrics**: FID score, LPIPS distance for synthetic samples

### Hyperparameter Tuning

#### Key Hyperparameters

1. **h_gamma**: Content injection strength
   - Start with 0.5 and adjust based on results
   - Higher values increase diversity but may reduce quality

2. **bias_conflict_ratio**: Ratio of synthetic to original samples
   - Typical range: 0.1-0.3
   - Balance between debiasing and maintaining performance

3. **Learning rate**: For classifier training
   - Start with 1e-3 and use learning rate scheduling
   - Monitor validation loss for early stopping

#### Tuning Strategy

1. **Grid search** over h_gamma values: [0.1, 0.3, 0.5, 0.7, 0.9]
2. **Cross-validation** for bias_conflict_ratio selection
3. **Ablation studies** for architectural choices

### Troubleshooting

#### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Poor Synthetic Sample Quality**
   - Adjust h_gamma values
   - Increase n_gen_step and n_inv_step
   - Check pretrained model quality

3. **Training Instability**
   - Reduce learning rate
   - Use gradient clipping
   - Check data preprocessing

#### Debugging Tips

1. **Monitor training curves** for convergence
2. **Visualize synthetic samples** to assess quality
3. **Check bias metrics** throughout training
4. **Validate on held-out test set**

### Extending DiffInject

#### Adding New Datasets

1. Create dataset configuration in `configs/`
2. Add dataset paths to `configs/paths_config.py`
3. Implement dataset-specific preprocessing in `train_classifier/`
4. Update model architecture if needed

#### Custom Loss Functions

The framework supports custom loss functions for different bias types:

```python
class CustomBiasLoss(nn.Module):
    def __init__(self, bias_type):
        super().__init__()
        self.bias_type = bias_type
        
    def forward(self, predictions, targets, bias_attributes):
        # Implement custom bias-aware loss
        pass
```

#### New Diffusion Models

To support different diffusion model architectures:

1. Implement the model interface in `models/`
2. Update h-space extraction in `diffusion_latent.py`
3. Add configuration options

## References

For more details on the underlying methods:

- DDIM: Denoising Diffusion Implicit Models
- H-space manipulation in diffusion models
- Generalized Cross Entropy loss
- Dataset bias mitigation techniques

## Contributing

When extending DiffInject:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Add configuration examples 