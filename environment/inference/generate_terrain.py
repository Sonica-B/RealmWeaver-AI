
import os
import sys
import argparse
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.models.vae import VAE as TerrainVAE

def parse_args():
    parser = argparse.ArgumentParser(description="Generate terrain using the trained VAE model")
    parser.add_argument("--model-path", type=str, 
                        default="outputs/terrain_vae/checkpoint-100/model.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--terrain-size", type=int, default=128, 
                        help="Size of the terrain grid")
    parser.add_argument("--latent-dim", type=int, default=64, 
                        help="Dimension of the latent space")
    parser.add_argument("--hidden-dim", type=int, default=256, 
                        help="Dimension of hidden layers")
    parser.add_argument("--num-samples", type=int, default=20, 
                        help="Number of terrain samples to generate")
    parser.add_argument("--output-dir", type=str, 
                        default="outputs/generated_terrain",
                        help="Directory to save generated terrain")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generate_terrain.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()
    logger = setup_logging()
    logger.info(f"Generating terrain with parameters: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    input_dim = args.terrain_size * args.terrain_size
    model = TerrainVAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        logger.warning(f"Model not found at {args.model_path}. Using untrained model.")
        
    model.to(device)
    model.eval()
    
    # Generate terrain samples
    logger.info(f"Generating {args.num_samples} terrain samples")
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(args.num_samples, args.latent_dim).to(device)
        terrain_samples = model.decode(z)
        terrain_samples = terrain_samples.view(-1, args.terrain_size, args.terrain_size).cpu().numpy()
    
    # Save terrain samples
    for i, terrain in enumerate(terrain_samples):
        # Save as numpy array
        np.save(output_dir / f"terrain_{i+1}.npy", terrain)
        
        # Save as image for visualization
        try:
            import matplotlib.pyplot as plt
            
            # As grayscale heightmap
            plt.figure(figsize=(10, 10))
            plt.imshow(terrain, cmap='terrain')
            plt.colorbar(label='Height')
            plt.title(f"Generated Terrain Sample {i+1}")
            plt.savefig(output_dir / f"terrain_{i+1}_heightmap.png")
            plt.close()
            
            # As 3D surface
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(0, args.terrain_size, 1)
            y = np.arange(0, args.terrain_size, 1)
            x, y = np.meshgrid(x, y)
            
            ax.plot_surface(x, y, terrain, cmap='terrain')
            ax.set_title(f"Generated Terrain Sample {i+1}")
            plt.savefig(output_dir / f"terrain_{i+1}_3d.png")
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping visualization.")
    
    logger.info(f"Saved {args.num_samples} terrain samples to {output_dir}")

if __name__ == "__main__":
    main()
