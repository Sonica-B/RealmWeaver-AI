import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from shared.utils.training_pipeline import TrainingPipeline

# Simple VAE model for testing
class SimpleVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple implementation for testing
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Placeholder implementation
        # Encode input to latent space
        encoded = self.encoder(x)
        
        # Split encoded representation into mean and log-variance
        mean, log_var = torch.chunk(encoded, 2, dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        # Decode latent representation back to input space
        decoded = self.decoder(z)
        return decoded, mean, log_var
        
if __name__ == "__main__":
    # Create simple model and config for testing
    model = SimpleVAE()
    config = {
        'output_dir': './outputs',
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10
    }
    
    # Create dummy dataset for testing
    dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 784)
    )
    
    # Initialize and test pipeline
    pipeline = TrainingPipeline(model, dataset, config)
    print("Training pipeline initialized successfully")
