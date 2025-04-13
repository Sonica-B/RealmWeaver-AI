import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# Add project root to path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.models.vae import VAE

def visualize_terrain(model_path, output_path, terrain_size=64, num_samples=4):
    """Visualize terrain samples from a trained VAE model."""
    # Load model

    input_dim = terrain_size * terrain_size
    model = VAE(input_dim=input_dim, latent_dim=64, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Generate samples

    with torch.no_grad():
        # Sample from latent space

        z = torch.randn(num_samples, 64)
        samples = model.decode(z)
        
    # Reshape and visualize

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*4, 4))
    for i, sample in enumerate(samples):
        terrain = sample.view(terrain_size, terrain_size).numpy()
        if num_samples > 1:
            ax = axes[i]
        else:
            ax = axes
        im = ax.imshow(terrain, cmap='terrain')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize terrain from VAE')
    parser.add_argument('--model-path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--output-path', type=str, default='terrain_samples.png', help='output image path')
    args = parser.parse_args()
    
    visualize_terrain(args.model_path, args.output_path)
