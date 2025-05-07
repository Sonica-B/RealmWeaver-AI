import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Configuration (match your training settings)
latent_dim = 100
num_classes = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "saved_models"  # Directory where models are saved
output_dir = "class_samples_cvaegans"  # Directory to save generated images
os.makedirs(output_dir, exist_ok=True)

# Define your Generator class (must match training)
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

# Load the trained generator
generator = Generator(latent_dim, num_classes).to(device)
generator.load_state_dict(torch.load(os.path.join(model_dir, "generator.pth")))
generator.eval()

# Generate and save one high-quality image per class
for cls in range(num_classes):
    # Create latent vector and one-hot label
    z = torch.randn(1, latent_dim, device=device)
    label = torch.zeros(1, num_classes, device=device)
    label[0, cls] = 1  # One-hot encoding
    
    # Generate image
    with torch.no_grad():
        generated_img = generator(z, label)
    
    # Save image (normalized from [-1,1] to [0,1])
    save_image(generated_img, 
               os.path.join(output_dir, f"class_{cls}.png"),
               normalize=True,
               padding=0)
    
    # Optional: Display the image
    plt.figure(figsize=(6, 6))
    img = generated_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
    plt.imshow(img)
    plt.title(f"Class {cls}")
    plt.axis('off')
    plt.show()

print(f"Successfully generated and saved one image per class in '{output_dir}' directory")