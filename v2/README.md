# CycleGAN with VGG Perceptual Loss for CIFAR-10 Domain Transfer

This repository contains an implementation of CycleGAN with VGG-based perceptual loss for image-to-image style transfer between CIFAR-10 domains. The implementation is based on PyTorch and is designed to work with Google Colab.

## Overview

CycleGAN is an unsupervised learning technique for image-to-image translation tasks where paired examples are not available. This implementation enhances the traditional CycleGAN by incorporating a perceptual loss based on VGG16 features to improve the quality of generated images.

The script allows you to:
- Extract specific class domains from CIFAR-10 dataset (e.g., horses and deer)
- Train CycleGAN models with perceptual loss to translate between domains
- Generate and visualize the translated images

## Key Features

- **Domain Preparation**: Automatically extracts and organizes images from CIFAR-10 classes
- **Enhanced CycleGAN Architecture**: Includes residual blocks for better gradient flow
- **Perceptual Loss**: Utilizes VGG16 features to assess perceptual similarities
- **Visualization**: Generates comparison images showing original, translated, and cycle consistency results
- **Device Optimization**: Supports CUDA, MPS (Apple Silicon), and CPU devices

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)
- Google Colab (optional, for cloud execution)

## Usage

### 1. Mount Google Drive (if using Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Prepare CIFAR-10 Domains

```python
domain_dirs = prepare_cifar10_domains(
    root='/content/drive/MyDrive/CycleGAN_VGG/datasets',
    domain_a='horse',  # Choose any CIFAR-10 class
    domain_b='deer'    # Choose any other CIFAR-10 class
)
```

Available CIFAR-10 classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'

### 3. Train the CycleGAN Model

```python
G_AB, G_BA = train_cyclegan(
    domain_dirs['domain_a_dir'],
    domain_dirs['domain_b_dir'],
    epochs=200,             # Number of training epochs
    batch_size=64,          # Batch size for training
    lr=0.0002,              # Learning rate
    decay_epoch=100,        # Epoch to start learning rate decay
    sample_interval=50      # Interval for saving sample images
)
```

### 4. Generate Transformed Images

```python
generate_images(
    G_AB,                         # Generator A→B
    G_BA,                         # Generator B→A
    domain_dirs['domain_a_dir'],  # Directory for domain A test images
    domain_dirs['domain_b_dir'],  # Directory for domain B test images
    num_images=10                 # Number of images to generate
)
```

## Model Architecture

### Generator
- Downsampling: Initial convolutional layer followed by two downsampling blocks
- Transformation: Six residual blocks for feature transformation
- Upsampling: Two upsampling blocks to restore original resolution
- Output: tanh activation for normalized pixel values

### Discriminator
- PatchGAN architecture for evaluating patches of the image
- Four convolutional layers with increasing filter sizes
- Instance normalization and LeakyReLU activations

### Loss Functions
- Adversarial Loss: MSE loss for aligning generated images with target distribution
- Cycle Consistency Loss: L1 loss to ensure cycle consistency
- Identity Loss: L1 loss to preserve color and content
- Perceptual Loss: Feature-based loss using VGG16 to improve realism

## Output Directory Structure

```
/content/drive/MyDrive/CycleGAN_VGG/
├── datasets/               # CIFAR-10 extracted domains
│   ├── cifar10_horse/
│   │   ├── trainA/
│   │   └── testA/
│   └── cifar10_deer/
│       ├── trainB/
│       └── testB/
├── images/                 # Training progress visualizations
├── checkpoints/            # Saved model weights
└── results/                # Final generated images
```

## Notes

- Training time can be significant. Consider using a GPU environment.
- The implementation uses a replay buffer to stabilize GAN training.
- The perceptual loss weight can be adjusted via the `lambda_perceptual` parameter.
- Change the `n_residual_blocks` parameter to adjust the generator's capacity.

## Example Results

After training, you can view the results in:
- `/content/drive/MyDrive/CycleGAN_VGG/images/` for progress visualization
- `/content/drive/MyDrive/CycleGAN_VGG/results/` for final transformations

## License

[MIT License]

## Acknowledgements

This implementation is inspired by the original CycleGAN paper (Zhu et al., 2017) and incorporates perceptual loss as proposed by Johnson et al. (2016).
