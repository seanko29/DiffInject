# DiffInject Path Configuration Template
# Copy this file to paths_config.py and update with your actual paths

import os

# Dataset paths - Update these with your actual dataset locations
DATASET_PATHS = {
    'ffhq': '/path/to/ffhq/dataset',
    'celeba': '/path/to/celeba/dataset', 
    'afhq': '/path/to/afhq/dataset',
    'metfaces': '/path/to/metfaces/dataset',
    'lsun_bedroom': '/path/to/lsun_bedroom/dataset',
    'lsun_church': '/path/to/lsun_church/dataset',
    'imagenet': '/path/to/imagenet/dataset',
    'cmnist': '/path/to/cmnist/dataset',
    'cifar10c': '/path/to/cifar10c/dataset',
    'bffhq': '/path/to/bffhq/dataset',
    'bar': '/path/to/bar/dataset',
}

# Pretrained diffusion model paths - Download these models and update paths
PRETRAINED_MODELS = {
    'ffhq': '/path/to/ffhq_model.pt',
    'celeba': '/path/to/celeba_model.pt',
    'afhq': '/path/to/afhq_model.pt',
    'metfaces': '/path/to/metfaces_model.pt',
    'lsun_bedroom': '/path/to/lsun_bedroom_model.pt',
    'lsun_church': '/path/to/lsun_church_model.pt',
    'imagenet': '/path/to/imagenet_model.pt',
}

# Model checkpoint names (used internally)
MODEL_NAMES = {
    'ffhq': 'ffhq_p2.pt',
    'celeba': 'celeba_p2.pt',
    'afhq': 'afhq_p2.pt',
    'metfaces': 'metfaces_p2.pt',
    'lsun_bedroom': 'lsun_bedroom_p2.pt',
    'lsun_church': 'lsun_church_p2.pt',
    'imagenet': 'imagenet_p2.pt',
}

# Test image directories - Create these directories and add your test images
TEST_IMAGE_PATHS = {
    'ffhq': {
        'contents': './test_images/ffhq/contents',
        'styles': './test_images/ffhq/styles',
    },
    'celeba': {
        'contents': './test_images/celeba/contents',
        'styles': './test_images/celeba/styles',
    },
    'afhq': {
        'contents': './test_images/afhq/contents',
        'styles': './test_images/afhq/styles',
    },
    'bffhq': {
        'contents': './test_images/bffhq/contents',
        'styles': './test_images/bffhq/styles',
    },
}

# Output directories - These will be created automatically
OUTPUT_PATHS = {
    'results': './results',
    'bin': './bin',
    'runs': './runs',
    'checkpoints': './checkpoints',
}

# Ensure all directories exist
def create_directories():
    """Create necessary directories if they don't exist"""
    for path_dict in [DATASET_PATHS, OUTPUT_PATHS]:
        for key, path in path_dict.items():
            if isinstance(path, str):
                os.makedirs(path, exist_ok=True)
            elif isinstance(path, dict):
                for subkey, subpath in path.items():
                    os.makedirs(subpath, exist_ok=True)

# Create directories when this module is imported
create_directories() 