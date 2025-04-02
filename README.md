# CycleGANExperiments

A comprehensive framework for exploring CycleGAN-based unpaired image-to-image translation across multiple domains and datasets.

![CycleGAN Example](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_1000_batch_60.png)

## Overview

CycleGANExperiments provides implementations of Cycle-Consistent Adversarial Networks (CycleGANs) for transforming images from one domain to another without requiring paired training data. This repository includes:

- Multiple dataset support (CIFAR-10, custom datasets, etc.)
- Modular architecture for easy experimentation
- Training monitoring and visualization
- Pre-trained models for quick testing
- Extensive configuration options

## Features

- **Multiple Dataset Support**: Works with CIFAR-10, ImageNet, custom datasets, and more
- **Flexible Architecture**: Easily swap generator and discriminator architectures
- **Training Tools**: Progress visualization, checkpoint saving, and learning rate scheduling
- **Inference Support**: Generate translations from trained models
- **Metrics**: Built-in evaluation metrics for generated images
- **Visualization**: Sample generation during training for progress monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CycleGANExperiments.git
cd CycleGANExperiments

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- matplotlib
- pillow
- tqdm

## Quick Start

### Training with CIFAR-10

```bash
python train.py --dataset cifar10 --domain_a horse --domain_b deer --epochs 200
```

### Training with Custom Datasets

```bash
python train.py --dataset custom --domain_a_path /path/to/domain_a --domain_b_path /path/to/domain_b --image_size 256 --epochs 300
```

### Generating Images with Pre-trained Models

```bash
python generate.py --model_path checkpoints/G_AB_epoch_200.pth --input_path test_images --output_path results
```

## Dataset Preparation

### CIFAR-10

The script automatically downloads and prepares CIFAR-10 data. Just specify the classes you want to use:

```bash
python prepare_data.py --dataset cifar10 --domain_a horse --domain_b deer
```

### Custom Datasets

For custom datasets, organize your images in the following structure:

```
datasets/
├── custom_dataset_name/
│   ├── trainA/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── trainB/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── testA/
│   │   ├── image1.jpg
│   │   └── ...
│   └── testB/
│       ├── image1.jpg
│       └── ...
```

Then run:

```bash
python prepare_data.py --dataset custom --root datasets/custom_dataset_name
```

## Architecture

CycleGANExperiments implements the architecture from the original CycleGAN paper with some modifications:

- Two generator networks: G_AB (A→B) and G_BA (B→A)
- Two discriminator networks: D_A and D_B
- Loss functions: adversarial loss, cycle consistency loss, and identity loss

## Configuration Options

Customize your experiments with numerous configuration options:

```bash
python train.py --help
```

Key parameters include:
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--decay_epoch`: Epoch to start learning rate decay
- `--n_residual_blocks`: Number of residual blocks in generators
- `--lambda_cycle`: Weight of cycle consistency loss
- `--lambda_identity`: Weight of identity loss

## Results

### Example Transformations

- Horse ↔ Deer
- And many more!

Model outputs are saved to the `results/` directory. Training progress images are saved to the `images/` directory.

## Model Zoo

Pre-trained models are available in the `models/` directory:

- `horse2deer_200.pth`: Horse to Deer transformation, trained for 1000 epochs
- (Add your own models)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Original CycleGAN Paper](https://arxiv.org/abs/1703.10593) - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- [PyTorch](https://pytorch.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
