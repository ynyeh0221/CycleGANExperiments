import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils, datasets, models
from PIL import Image
import matplotlib.pyplot as plt

# If using Colab
from google.colab import drive
drive.mount('/content/drive')

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Check if GPU is available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)

# STL-10 class labels
stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


# Create dataset directories for STL-10
def prepare_stl10_domains(root='/content/drive/MyDrive/CycleGAN/datasets', domain_a='horse', domain_b='deer'):
    """
    Prepares STL-10 dataset for CycleGAN by selecting two domains.

    Args:
        root (str): Root directory for datasets
        domain_a (str): First domain (must be one of STL-10 classes)
        domain_b (str): Second domain (must be one of STL-10 classes)

    Returns:
        dict: Paths to domain directories
    """
    if domain_a not in stl10_classes or domain_b not in stl10_classes:
        raise ValueError(f"Domains must be one of {stl10_classes}")

    # Create necessary directories
    domain_a_idx = stl10_classes.index(domain_a)
    domain_b_idx = stl10_classes.index(domain_b)

    os.makedirs(root, exist_ok=True)

    # Create domain directories
    domain_a_dir = os.path.join(root, f'stl10_{domain_a}')
    domain_b_dir = os.path.join(root, f'stl10_{domain_b}')

    train_a_dir = os.path.join(domain_a_dir, 'trainA')
    test_a_dir = os.path.join(domain_a_dir, 'testA')
    train_b_dir = os.path.join(domain_b_dir, 'trainB')
    test_b_dir = os.path.join(domain_b_dir, 'testB')

    for dir_path in [train_a_dir, test_a_dir, train_b_dir, test_b_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Created domain directories for {domain_a} and {domain_b}")

    # Download STL-10 dataset
    print("Downloading STL-10 dataset (if not already downloaded)...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.STL10(root=root, split='train', download=True, transform=transform)
    test_dataset = datasets.STL10(root=root, split='test', download=True, transform=transform)

    # Extract images for domain A (train)
    domain_a_indices_train = [i for i, (_, label) in enumerate(train_dataset) if label == domain_a_idx]
    for i, idx in enumerate(domain_a_indices_train):
        img, _ = train_dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(train_a_dir, f'{i:05d}.png'))

    # Extract images for domain A (test)
    domain_a_indices_test = [i for i, (_, label) in enumerate(test_dataset) if label == domain_a_idx]
    for i, idx in enumerate(domain_a_indices_test):
        img, _ = test_dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(test_a_dir, f'{i:05d}.png'))

    # Extract images for domain B (train)
    domain_b_indices_train = [i for i, (_, label) in enumerate(train_dataset) if label == domain_b_idx]
    for i, idx in enumerate(domain_b_indices_train):
        img, _ = train_dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(train_b_dir, f'{i:05d}.png'))

    # Extract images for domain B (test)
    domain_b_indices_test = [i for i, (_, label) in enumerate(test_dataset) if label == domain_b_idx]
    for i, idx in enumerate(domain_b_indices_test):
        img, _ = test_dataset[idx]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(test_b_dir, f'{i:05d}.png'))

    print(f"Domain A (train): {len(domain_a_indices_train)} images")
    print(f"Domain A (test): {len(domain_a_indices_test)} images")
    print(f"Domain B (train): {len(domain_b_indices_train)} images")
    print(f"Domain B (test): {len(domain_b_indices_test)} images")

    return {
        'domain_a_dir': domain_a_dir,
        'domain_b_dir': domain_b_dir
    }


# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, domain='A', mode='train', transform=None):
        self.transform = transform
        self.files = []

        try:
            if mode == 'train':
                path = os.path.join(root_dir, f'train{domain}')
                if os.path.exists(path):
                    self.files = os.listdir(path)
                    self.files = [os.path.join(path, x) for x in self.files]
            else:
                path = os.path.join(root_dir, f'test{domain}')
                if os.path.exists(path):
                    self.files = os.listdir(path)
                    self.files = [os.path.join(path, x) for x in self.files]

            # Filter out any non-image files
            self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.files = []

        self.size = len(self.files)
        print(f"Loaded {self.size} images for {mode}{domain}")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.size == 0:
            img = Image.new('RGB', (96, 96), color='black')
            if self.transform:
                img = self.transform(img)
            return img

        img_path = self.files[index % self.size]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (96, 96), color='black')
            if self.transform:
                img = self.transform(img)
            return img


# Define the weights initialization method
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


# Generator Network
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=6):
        super(Generator, self).__init__()
        # Initial convolutional layer
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# Buffer of previously generated samples
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)


# Learning rate scheduler
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


# -----------------------------------------
# VGG Feature Extractor for Perceptual Loss
# -----------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # Use pretrained VGG16 and extract features up to layer 16 (e.g., relu3_3)
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x is in range [-1, 1]. Convert to [0, 1]
        x = (x + 1) / 2
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.features(x)


# Modified training function including perceptual cycle loss
def train_cyclegan(domain_a_dir, domain_b_dir, epochs=100, batch_size=4, lr=0.0002, decay_epoch=50,
                   sample_interval=100):
    # Data transforms; note STL-10 images are 96x96 so we use them as-is
    transforms_ = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Create datasets and dataloaders
    trainA = ImageDataset(domain_a_dir, domain='A', mode='train', transform=transforms.Compose(transforms_))
    trainB = ImageDataset(domain_b_dir, domain='B', mode='train', transform=transforms.Compose(transforms_))
    testA = ImageDataset(domain_a_dir, domain='A', mode='test', transform=transforms.Compose(transforms_))
    testB = ImageDataset(domain_b_dir, domain='B', mode='test', transform=transforms.Compose(transforms_))

    if len(trainA) == 0 or len(trainB) == 0:
        print("Error: Training datasets are empty.")
        sys.exit(1)

    print(f"Dataset sizes: TrainA: {len(trainA)}, TrainB: {len(trainB)}, TestA: {len(testA)}, TestB: {len(testB)}")

    dataloader_A = DataLoader(trainA, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_B = DataLoader(trainB, batch_size=batch_size, shuffle=True, num_workers=4)

    os.makedirs("/content/drive/MyDrive/CycleGAN/samples", exist_ok=True)

    # Loss criteria
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Initialize networks
    G_AB = Generator(n_residual_blocks=6).to(device)
    G_BA = Generator(n_residual_blocks=6).to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Instantiate VGG extractor for perceptual loss
    vgg_extractor = VGGFeatureExtractor().to(device)
    vgg_extractor.eval()
    lambda_perceptual = 0.5  # Weight for perceptual loss

    # Optimizers
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    lambda_func_G = LambdaLR(epochs, 0, decay_epoch)
    lambda_func_D_A = LambdaLR(epochs, 0, decay_epoch)
    lambda_func_D_B = LambdaLR(epochs, 0, decay_epoch)

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: lambda_func_G.step(epoch)
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: lambda_func_D_A.step(epoch)
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: lambda_func_D_B.step(epoch)
    )

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            batch_size_curr = min(real_A.size(0), real_B.size(0))
            real_A = real_A[:batch_size_curr].to(device)
            real_B = real_B[:batch_size_curr].to(device)

            valid = torch.ones((batch_size_curr, 1), device=device, requires_grad=False)
            fake = torch.zeros((batch_size_curr, 1), device=device, requires_grad=False)

            # ------------------
            # Train Generators
            # ------------------
            G_AB.train()
            G_BA.train()
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle consistency with pixel and perceptual loss
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)

            loss_cycle_A_pixel = criterion_cycle(recov_A, real_A)
            loss_cycle_B_pixel = criterion_cycle(recov_B, real_B)
            loss_cycle_A_perceptual = criterion_cycle(vgg_extractor(recov_A), vgg_extractor(real_A))
            loss_cycle_B_perceptual = criterion_cycle(vgg_extractor(recov_B), vgg_extractor(real_B))

            loss_cycle_A = loss_cycle_A_pixel + lambda_perceptual * loss_cycle_A_perceptual
            loss_cycle_B = loss_cycle_B_pixel + lambda_perceptual * loss_cycle_B_perceptual
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total generator loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # ----------------------
            # Train Discriminator A
            # ----------------------
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A.detach())
            loss_fake = criterion_GAN(D_A(fake_A_), fake)
            loss_D_A = (loss_real + loss_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ----------------------
            # Train Discriminator B
            # ----------------------
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B.detach())
            loss_fake = criterion_GAN(D_B(fake_B_), fake)
            loss_D_B = (loss_real + loss_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            if i % 50 == 0:
                print(
                    f"[Epoch {epoch + 1}/{epochs}] "
                    f"[Batch {i}/{min(len(dataloader_A), len(dataloader_B))}] "
                    f"[D_A loss: {loss_D_A.item():.4f}] "
                    f"[D_B loss: {loss_D_B.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}, "
                    f"adv: {loss_GAN.item():.4f}, "
                    f"cycle: {loss_cycle.item():.4f}, "
                    f"identity: {loss_identity.item():.4f}]"
                )

            if i % sample_interval == 0 and len(testA) > 0 and len(testB) > 0:
                G_AB.eval()
                G_BA.eval()
                os.makedirs("/content/drive/MyDrive/CycleGAN/images", exist_ok=True)

                def denorm(x):
                    return (x * 0.5 + 0.5).clamp(0, 1)

                real_A_test = next(iter(DataLoader(testA, batch_size=5))).to(device)
                real_B_test = next(iter(DataLoader(testB, batch_size=5))).to(device)

                fake_B_test = G_AB(real_A_test)
                fake_A_test = G_BA(real_B_test)

                fig, axs = plt.subplots(2, 3, figsize=(12, 8))
                axs[0, 0].imshow(denorm(real_A_test[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[0, 0].set_title("Original A")
                axs[0, 0].axis("off")
                axs[0, 1].imshow(denorm(fake_B_test[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[0, 1].set_title("Translated A->B")
                axs[0, 1].axis("off")
                axs[0, 2].imshow(denorm(G_BA(fake_B_test)[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[0, 2].set_title("Cycle A->B->A")
                axs[0, 2].axis("off")
                axs[1, 0].imshow(denorm(real_B_test[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[1, 0].set_title("Original B")
                axs[1, 0].axis("off")
                axs[1, 1].imshow(denorm(fake_A_test[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[1, 1].set_title("Translated B->A")
                axs[1, 1].axis("off")
                axs[1, 2].imshow(denorm(G_AB(fake_A_test)[0]).cpu().permute(1, 2, 0).detach().numpy())
                axs[1, 2].set_title("Cycle B->A->B")
                axs[1, 2].axis("off")
                plt.tight_layout()
                plt.savefig(f"/content/drive/MyDrive/CycleGAN/images/epoch_{epoch + 1}_batch_{i}.png")
                plt.close()

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if (epoch + 1) % 50 == 0:
            os.makedirs("/content/drive/MyDrive/CycleGAN/checkpoints", exist_ok=True)
            torch.save(G_AB.state_dict(), f"/content/drive/MyDrive/CycleGAN/checkpoints/G_AB_epoch_{epoch + 1}.pth")
            torch.save(G_BA.state_dict(), f"/content/drive/MyDrive/CycleGAN/checkpoints/G_BA_epoch_{epoch + 1}.pth")
            torch.save(D_A.state_dict(), f"/content/drive/MyDrive/CycleGAN/checkpoints/D_A_epoch_{epoch + 1}.pth")
            torch.save(D_B.state_dict(), f"/content/drive/MyDrive/CycleGAN/checkpoints/D_B_epoch_{epoch + 1}.pth")

    return G_AB, G_BA


# Function to generate transformed images using trained generators
def generate_images(G_AB, G_BA, test_A_path, test_B_path, num_images=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_A = ImageDataset(test_A_path, domain='A', mode='test', transform=transform)
    test_B = ImageDataset(test_B_path, domain='B', mode='test', transform=transform)

    if len(test_A) == 0 or len(test_B) == 0:
        print("Warning: Test datasets are empty. Cannot generate images.")
        return

    os.makedirs("/content/drive/MyDrive/CycleGAN/results", exist_ok=True)

    for i in range(min(num_images, len(test_A))):
        real_A = test_A[i].unsqueeze(0).to(device)
        fake_B = G_AB(real_A)
        real_A = (real_A * 0.5 + 0.5).clamp(0, 1)
        fake_B = (fake_B * 0.5 + 0.5).clamp(0, 1)
        utils.save_image(real_A, f"/content/drive/MyDrive/CycleGAN/results/real_A_{i}.png")
        utils.save_image(fake_B, f"/content/drive/MyDrive/CycleGAN/results/fake_B_{i}.png")

    for i in range(min(num_images, len(test_B))):
        real_B = test_B[i].unsqueeze(0).to(device)
        fake_A = G_BA(real_B)
        real_B = (real_B * 0.5 + 0.5).clamp(0, 1)
        fake_A = (fake_A * 0.5 + 0.5).clamp(0, 1)
        utils.save_image(real_B, f"/content/drive/MyDrive/CycleGAN/results/real_B_{i}.png")
        utils.save_image(fake_A, f"/content/drive/MyDrive/CycleGAN/results/fake_A_{i}.png")


# Main execution
if __name__ == "__main__":
    try:
        # Define domains from STL-10 classes
        domain_a = 'horse'
        domain_b = 'deer'
        print(f"Setting up CycleGAN for {domain_a} ↔ {domain_b} style transfer using STL-10")
        domain_dirs = prepare_stl10_domains(domain_a=domain_a, domain_b=domain_b)
        G_AB, G_BA = train_cyclegan(
            domain_dirs['domain_a_dir'],
            domain_dirs['domain_b_dir'],
            epochs=1000,
            batch_size=16,
            lr=0.0002,
            decay_epoch=50,
            sample_interval=50
        )
        generate_images(G_AB, G_BA, domain_dirs['domain_a_dir'], domain_dirs['domain_b_dir'], num_images=10)
        print("Training and generation complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the script and dataset, then try again.")
