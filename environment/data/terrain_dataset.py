import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SimplifiedTerrainDataset(Dataset):
    """Generates simplified terrain height maps for training."""
    
    def __init__(self, size=64, num_samples=1000):
        self.size = size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a simplified height map using perlin-like noise
        height_map = self._generate_height_map()
        # Flatten for VAE input
        return torch.from_numpy(height_map).float().view(-1)
    
    def _generate_height_map(self):
        # Simple procedural terrain generation
        x = np.linspace(0, 5, self.size)
        y = np.linspace(0, 5, self.size)
        xv, yv = np.meshgrid(x, y)
        
        # Generate some noise patterns
        z = np.sin(xv) + np.cos(yv)
        z += np.random.normal(0, 0.1, (self.size, self.size))
        
        # Normalize to [0, 1]
        z = (z - z.min()) / (z.max() - z.min())
        
        return z
