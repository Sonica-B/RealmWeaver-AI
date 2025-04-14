import os
import torch
import logging
from pathlib import Path

def setup_environment():
    """Set up project directory structure and logging"""
    # Create directory structure
    project_dirs = [
        "environment/data",
        "environment/models",
        "environment/training",
        "environment/inference",
        "character/data",
        "character/models",
        "shared/utils",
        "outputs"
    ]
    
    for directory in project_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("gameworldgen/logs.log"),
            logging.StreamHandler()
        ]
    )
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    if device == "cuda":
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return device

# Basic VAE implementation for terrain generation
class TerrainVAE(torch.nn.Module):
    def __init__(self, input_dim=4096, latent_dim=64, hidden_dim=256):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var