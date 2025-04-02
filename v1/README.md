# CIFAR-10 CycleGAN Experiment

A PyTorch implementation of CycleGAN for unpaired image-to-image translation between CIFAR-10 classes.

## Description

This project implements CycleGAN architecture to perform style transfer between two domains from the CIFAR-10 dataset. The default implementation transforms between 'horse' and 'deer' classes, but can be adapted to work with any pair of CIFAR-10 classes.

## Features

- Automatic dataset preparation from CIFAR-10
- CycleGAN implementation with identity, adversarial, and cycle consistency losses
- Support for training on GPU, MPS (Apple Silicon), or CPU
- Visualization of training progress
- Sample generation from trained models

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cifar10-cyclegan.git
cd cifar10-cyclegan
```

2. Install the required packages:
```bash
pip install torch torchvision numpy matplotlib pillow
```

## Usage

Run the script with default parameters:
```bash
python cyclegan_cifar10.py
```

To modify the domains, edit the following lines in the script:
```python
# Main execution
if __name__ == "__main__":
    # Define domains from CIFAR-10 classes
    domain_a = 'horse'  # First domain
    domain_b = 'deer'   # Second domain
```

Available CIFAR-10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## How It Works

1. **Data Preparation**: The script extracts images from two CIFAR-10 classes and organizes them into domain directories.
2. **Network Architecture**: 
   - Two generator networks (G_AB and G_BA) translate images between domains
   - Two discriminator networks (D_A and D_B) distinguish between real and fake images
3. **Training Process**:
   - Identity loss: Ensures generators preserve images from their target domain
   - Adversarial loss: Encourages generators to produce realistic images
   - Cycle consistency loss: Enforces that translating an image to the other domain and back produces the original image
4. **Image Generation**: After training, the models can transform new images between domains

## Output

The script saves:
- Sample images during training in the `images/` directory
- Model checkpoints in the `checkpoints/` directory
- Final generated images in the `results/` directory

## Training Parameters

You can adjust the following parameters in the `train_cyclegan` function:
- `epochs`: Number of training epochs (default: 1000)
- `batch_size`: Batch size for training (default: 64)
- `lr`: Learning rate (default: 0.0002)
- `decay_epoch`: Epoch to start learning rate decay (default: 50)
- `sample_interval`: Interval between saving sample images (default: 30)

## Example Results

After training, you can find:
- Original and translated images in the `results/` directory
- Visual comparison of translation quality and cycle consistency

| | Example Result|
|-----------------|---------------|
| Epoch 100 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_100_batch_60.png) |
| Epoch 300 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_300_batch_60.png) |
| Epoch 500 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_500_batch_60.png) |
| Epoch 700 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_700_batch_60.png) |
| Epoch 900 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_900_batch_60.png) |
| Epoch 1000 | ![](https://github.com/ynyeh0221/CycleGANExperiments/blob/main/v1/output/images/epoch_1000_batch_60.png) |

## Limitations

- CIFAR-10 images are small (32x32 pixels), so the generated images will be low resolution
- Training may take several hours depending on your hardware
- Results may vary based on the chosen domains and similarity between classes

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593) - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
