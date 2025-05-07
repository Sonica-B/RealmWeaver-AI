import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # <-- Added for plotting

# -------------------------------
#       Model Definitions
# -------------------------------

class Generator(nn.Module):
    def __init__(self, num_classes, embedding_dim, latent_dim, img_size, channels):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.init_size = img_size // 16
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embedding_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, embedding_dim, img_size, channels):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        ds_size = img_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size + embedding_dim, 1),
        )

    def forward(self, img, labels):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        label_input = self.label_emb(labels)
        d_in = torch.cat((out, label_input), -1)
        validity = self.adv_layer(d_in)
        return validity

# -------------------------------
#         Utility Functions
# -------------------------------

def get_dataloader(data_dir, img_size, channels, batch_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),                   # Aggressive random crop & resize
        transforms.RandomHorizontalFlip(p=0.5),                                     # Horizontal flip
        transforms.RandomVerticalFlip(p=0.2),                                       # Vertical flip (if orientation doesn't matter)
        transforms.RandomAffine(
            degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),         # Affine transformations
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),                 # Stronger color jitter
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),                  # Perspective distortion
        transforms.ToTensor(),
        transforms.Normalize([0.5]*channels, [0.5]*channels)
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Class-to-index mapping:", dataset.class_to_idx)
    return dataloader, dataset.class_to_idx


def generate_images_with_trained_model(
    generator_path,
    latent_dim,
    num_classes,
    embedding_dim,
    img_size,
    channels,
    output_dir=".",
    device=None
):
    """Loads the trained generator and saves one generated image per class."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator(num_classes, embedding_dim, latent_dim, img_size, channels).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(num_classes, latent_dim, device=device)
        labels = torch.arange(num_classes, device=device)
        imgs = generator(z, labels)
        for idx, img in enumerate(imgs):
            save_image(img, os.path.join(output_dir, f"class_{idx}.png"), normalize=True)

# -------------------------------
#         Training Loop
# -------------------------------

def train_cgan(
    data_dir='Data',
    latent_dim=20,
    num_classes=6,
    embedding_dim=20,
    img_size=64,
    channels=3,
    batch_size=8,
    num_epochs=100,
    lr=0.0002,
    b1=0.5,
    b2=0.999,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    dataloader, class_to_idx = get_dataloader(data_dir, img_size, channels, batch_size)

    # Models
    generator = Generator(num_classes, embedding_dim, latent_dim, img_size, channels).to(device)
    discriminator = Discriminator(num_classes, embedding_dim, img_size, channels).to(device)

    # Loss and optimizers
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(b1, b2))

    d_loss_lst = []
    g_loss_lst = []

    os.makedirs("generated_images_per_epochs_cgans", exist_ok=True)

    # Training loop
    for epoch in range(1, num_epochs+1):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (imgs, labels) in enumerate(tqdm(dataloader)):
            batch_size_curr = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            valid = torch.ones(batch_size_curr, 1, device=device)
            fake = torch.zeros(batch_size_curr, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_G.zero_grad()
            z = torch.randn(batch_size_curr, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)
            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
        # Average loss for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_loss_lst.append(avg_g_loss)
        d_loss_lst.append(avg_d_loss)

        generator.eval()
        with torch.no_grad():
            z = torch.randn(num_classes, latent_dim, device=device)
            labels = torch.arange(num_classes, device=device)
            gen_imgs = generator(z, labels)
            save_image(gen_imgs, os.path.join("generated_images_per_epochs_cgans", f"generated_epoch_{epoch}.png"), nrow=num_classes, normalize=True)
        generator.train()

    # Save the final trained models
    save_dir = "saved_models"   
    os.makedirs(save_dir, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(save_dir, "generator_cgan.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator_cgan.pth"))
    print("Final models saved as generator_cgan.pth and discriminator_cgan.pth")
    print("Training finished.")

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(g_loss_lst, label='Generator Loss')
    plt.plot(d_loss_lst, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cgan_loss_plot.png')
    plt.show()

# -------------------------------
#         Main Block
# -------------------------------

if __name__ == "__main__":
    train_cgan(
        data_dir='Data',
        latent_dim=20,
        num_classes=6,
        embedding_dim=20,
        img_size=64,
        channels=3,
        batch_size=8,
        num_epochs=100,
        lr=0.0002,
        b1=0.5,
        b2=0.999,
    )
