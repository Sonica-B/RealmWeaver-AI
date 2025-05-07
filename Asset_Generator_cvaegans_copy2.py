import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms, utils

# -------------------------
# Data Loading Function
# -------------------------
def get_dataloader(data_dir, img_size, channels, batch_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*channels, [0.5]*channels)
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Class-to-index mapping:", dataset.class_to_idx)
    return dataloader, dataset.class_to_idx

# -------------------------
# Model Components (same as before)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.img_fc = nn.Linear(512*16*16, 256)
        self.label_fc = nn.Linear(num_classes, 256)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x, labels):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x_img = self.img_fc(x)
        x_lbl = self.label_fc(labels)
        x_combined = torch.cat([x_img, x_lbl], dim=1)
        return self.fc_mu(x_combined), self.fc_logvar(x_combined)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 512*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        x = self.fc(x).view(-1, 512, 16, 16)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(512*16*16, 1)

    def forward(self, x, labels):
        labels = labels.view(-1, labels.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        x = self.conv(x).flatten(start_dim=1)
        return self.fc(x)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc = nn.Linear(512*16*16, num_classes)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        return self.fc(x)

# -------------------------
# Loss Functions (same as before)
# -------------------------
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reconstruction_loss(x, x_recon):
    return nn.MSELoss()(x_recon, x)

def adversarial_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)

def classification_loss(pred, target):
    return nn.CrossEntropyLoss()(pred, target)

def feature_matching_loss(f_real, f_fake):
    return torch.mean((f_real.mean(dim=0) - f_fake.mean(dim=0))**2)

# -------------------------
# Visualization Function
# -------------------------
def visualize_samples(generator, num_classes, latent_dim, device, epoch, samples_per_class=3, img_scale=4):
    generator.eval()
    
    # Create figure with adjusted size
    fig = plt.figure(figsize=(img_scale*num_classes, img_scale*samples_per_class))
    plt.suptitle(f"Generated Samples - Epoch {epoch}", y=1.05, fontsize=16)
    
    # Generate and plot samples for each class
    for cls in range(num_classes):
        with torch.no_grad():
            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.eye(num_classes, device=device)[[cls]*samples_per_class]
            samples = generator(z, labels).cpu()
            
        # Plot each sample individually in a grid
        for i in range(samples_per_class):
            ax = fig.add_subplot(samples_per_class, num_classes, 1 + cls + i*num_classes)
            img = samples[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
            ax.imshow(img)
            if i == 0:
                ax.set_title(f"Class {cls}", fontsize=12)
            ax.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs("generated_images_per_epochs_cvaegans", exist_ok=True)
    plt.savefig(f"generated_images_per_epochs_cvaegans/epoch_{epoch:03d}.png", 
               bbox_inches='tight', 
               dpi=100,  # Increase DPI for better quality
               pad_inches=0.1)
    plt.close()
    generator.train()

# -------------------------
# Training Function (updated)
# -------------------------
def train_cvaegan(models, dataloader, num_epochs, device, num_classes, latent_dim):
    encoder, generator, discriminator, classifier = models
    opt_E = optim.Adam(encoder.parameters(), lr=1e-4)
    opt_G = optim.Adam(generator.parameters(), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    opt_C = optim.Adam(classifier.parameters(), lr=1e-4)

    recon_list, kl_list, adv_list, cls_list, fm_list, gen_list, disc_list = [], [], [], [], [], [], []

    for epoch in range(1, num_epochs+1):
        ep_recon = ep_kl = ep_adv = ep_cls = ep_fm = ep_gen = ep_disc = 0
        batches = 0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels_idx = labels.to(device)
            labels_onehot = torch.eye(num_classes, device=device)[labels_idx]
            B = imgs.size(0)

            # ================ Forward Pass ================
            # Encode
            mu, logvar = encoder(imgs, labels_onehot)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

            # Generate
            recon_imgs = generator(z, labels_onehot)
            z_prior = torch.randn(B, latent_dim, device=device)
            fake_imgs = generator(z_prior, labels_onehot)

            # ================ Discriminator Update ================
            real_val = discriminator(imgs, labels_onehot)
            fake_val = discriminator(fake_imgs.detach(), labels_onehot)
            d_loss = adversarial_loss(real_val, torch.ones_like(real_val)) + \
                     adversarial_loss(fake_val, torch.zeros_like(fake_val))
            
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ================ Classifier Update ================
            real_cls = classifier(imgs)
            fake_cls = classifier(fake_imgs.detach())
            c_loss = classification_loss(real_cls, labels_idx) + classification_loss(fake_cls, labels_idx)
            
            opt_C.zero_grad()
            c_loss.backward()
            opt_C.step()

            # ================ Generator & Encoder Update ================
            fake_val2 = discriminator(fake_imgs, labels_onehot)
            fake_cls2 = classifier(fake_imgs)

            # Feature matching for discriminator
            labels_expanded = labels_onehot.view(-1, labels_onehot.size(1), 1, 1).expand(-1, -1, imgs.size(2), imgs.size(3))
            f_real_D = discriminator.conv(torch.cat([imgs, labels_expanded], dim=1))
            f_fake_D = discriminator.conv(torch.cat([fake_imgs, labels_expanded], dim=1))
            fm_D = feature_matching_loss(f_real_D, f_fake_D)

            # Feature matching for classifier
            f_real_C = classifier.conv(imgs)
            f_fake_C = classifier.conv(fake_imgs)
            fm_C = feature_matching_loss(f_real_C, f_fake_C)

            # Loss components
            recon_l = reconstruction_loss(imgs, recon_imgs)
            kl_l = kl_divergence(mu, logvar) / B
            adv_l = adversarial_loss(fake_val2, torch.ones_like(fake_val2))
            cls_l = classification_loss(fake_cls2, labels_idx)
            fm_l = fm_D + fm_C

            g_loss = recon_l + 0.0005*kl_l + adv_l + cls_l + 10*fm_l

            opt_E.zero_grad()
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            opt_E.step()

            # ================ Loss Tracking ================
            ep_recon += recon_l.item()
            ep_kl += kl_l.item()
            ep_adv += adv_l.item()
            ep_cls += cls_l.item()
            ep_fm += fm_l.item()
            ep_gen += g_loss.item()
            ep_disc += d_loss.item()
            batches += 1

        # ================ Epoch Logging ================
        recon_list.append(ep_recon / batches)
        kl_list.append(ep_kl / batches)
        adv_list.append(ep_adv / batches)
        cls_list.append(ep_cls / batches)
        fm_list.append(ep_fm / batches)
        gen_list.append(ep_gen / batches)
        disc_list.append(ep_disc / batches)
        
        print(f"Epoch {epoch}/{num_epochs} | "
              f"D: {disc_list[-1]:.4f} | G: {gen_list[-1]:.4f} | "
              f"Recon: {recon_list[-1]:.4f} | KL: {kl_list[-1]:.4f} | "
              f"Adv: {adv_list[-1]:.4f} | Cls: {cls_list[-1]:.4f} | "
              f"FM: {fm_list[-1]:.4f}")

        # Generate and visualize samples after each epoch
        # In your training loop, call it like this:
        visualize_samples(generator, num_classes, latent_dim, device, epoch, 
                        samples_per_class=5,  # Show 5 samples per class
                        img_scale=5)         # Make images larger

    return recon_list, kl_list, adv_list, cls_list, fm_list, gen_list, disc_list
    
# -------------------------
# Main Function (same as before)
# -------------------------
if __name__ == "__main__":
    # Configuration
    data_dir = "Data"
    save_dir = "saved_models"
    out_dir = "generated_samples"
    latent_dim = 100
    num_classes = 6
    img_size = 256
    batch_size = 8
    num_epochs = 65
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Initialization
    enc = Encoder(latent_dim, num_classes).to(device)
    gen = Generator(latent_dim, num_classes).to(device)
    dis = Discriminator(num_classes).to(device)
    clsf = Classifier(num_classes).to(device)

    # Data Loading
    dl, class_map = get_dataloader(data_dir, img_size, 3, batch_size)
    print("Classes:", class_map)

    # Training
    recon_l, kl_l, adv_l, cls_l, fm_l, gen_l, disc_l = train_cvaegan(
        [enc, gen, dis, clsf], dl, num_epochs, device, num_classes, latent_dim
    )

    # Final Visualization and Saving
    def plot_losses(recon_l, kl_l, adv_l, cls_l, fm_l, gen_l, disc_l, num_epochs):
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(12, 8))

        plt.subplot(2,1,1)
        plt.plot(epochs, recon_l, label="Reconstruction")
        plt.plot(epochs, kl_l,    label="KL Divergence")
        plt.plot(epochs, adv_l,   label="Adversarial")
        plt.plot(epochs, cls_l,   label="Classification")
        plt.plot(epochs, fm_l,    label="Feature Matching")
        plt.title("Generator Loss Components")
        plt.legend(); plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(epochs, gen_l,  label="Total Generator")
        plt.plot(epochs, disc_l, label="Discriminator")
        plt.title("Generator vs Discriminator Loss")
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig("cvaegans_loss_plot.png")
        plt.show()

    plot_losses(recon_l, kl_l, adv_l, cls_l, fm_l, gen_l, disc_l, num_epochs)
    
    def save_models(models, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        names = ["encoder.pth", "generator.pth", "discriminator.pth", "classifier.pth"]
        for m, n in zip(models, names):
            torch.save(m.state_dict(), os.path.join(save_dir, n))

    save_models([enc, gen, dis, clsf], save_dir)
    
    def generate_images(generator, num_classes, latent_dim, device, save_dir, n_samples=5):
        os.makedirs(save_dir, exist_ok=True)
        generator.eval()
        for cls in range(num_classes):
            z = torch.randn(n_samples, latent_dim, device=device)
            labels = torch.eye(num_classes, device=device)[[cls]*n_samples]
            with torch.no_grad():
                imgs = generator(z, labels)
            utils.save_image(imgs,
                            os.path.join(save_dir, f"class_{cls}.png"),
                            nrow=n_samples, normalize=True)
        generator.train()

    generate_images(gen, num_classes, latent_dim, device, out_dir)
    
    print("Training complete. Models saved and samples generated.")