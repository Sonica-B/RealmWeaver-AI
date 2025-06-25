#!/usr/bin/env python
# train_vae.py - Training script for the Terrain VAE model

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Add project root to path to allow imports across modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.models.vae import VAE
from environment.data.terrain_dataset import TerrainDataset
from shared.utils.training_pipeline import TerrainVAETrainer as TrainingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model for terrain generation")
    parser.add_argument("--terrain-size", type=int, default=64, help="Size of the terrain grid")
    parser.add_argument("--latent-dim", type=int, default=64, help="Dimension of the latent space")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Dimension of hidden layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--output-dir", type=str, default="gameworldgen/outputs/terrain_vae", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--visualize", action="store_true", help="Visualize results after training")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("gameworldgen/train_vae.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    args = parse_args()
    logger = setup_logging()
    logger.info(f"Starting terrain VAE training with parameters: {args}")
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    input_dim = args.terrain_size * args.terrain_size
    model = VAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    model.to(device)
    
    # Create dataset
    dataset = TerrainDataset(
        size=args.terrain_size,
        num_samples=args.num_samples
    )
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Configure training
    config = {
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_workers': 2 if torch.cuda.is_available() else 0,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'save_interval': args.save_interval,
        'device': device
    }
    
    # Create trainer
    trainer = TrainingPipeline(model, dataset, config)
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")
    
    # Visualize results if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                z = torch.randn(5, args.latent_dim).to(device)
                samples = model.decode(z)
                samples = samples.view(-1, args.terrain_size, args.terrain_size).cpu().numpy()
            
            # Plot samples
            fig, axes = plt.subplots(1, len(samples), figsize=(15, 3))
            for i, sample in enumerate(samples):
                axes[i].imshow(sample, cmap='terrain')
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "samples.png"))
            plt.close()
            
            logger.info(f"Saved visualization to {os.path.join(args.output_dir, 'samples.png')}")
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    main()