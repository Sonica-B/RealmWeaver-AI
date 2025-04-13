import numpy as np
import torch
from torch.utils.data import Dataset

class TerrainDataset(Dataset):
    """Generate procedural terrain heightmaps for training"""
    def __init__(self, size=64, num_samples=1000):
        self.size = size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a heightmap using perlin-like noise
        heightmap = self._generate_heightmap()
        return torch.from_numpy(heightmap).float().view(-1)
    
    def _generate_heightmap(self):
        # Simple procedural terrain generation
        x = np.linspace(0, 5, self.size)
        y = np.linspace(0, 5, self.size)
        xv, yv = np.meshgrid(x, y)
        
        # Generate noise patterns at different frequencies
        z1 = np.sin(xv*0.5) * np.cos(yv*0.5)
        z2 = np.sin(xv*2) * np.cos(yv*2) * 0.3
        z3 = np.random.normal(0, 0.1, (self.size, self.size))
        
        # Combine them
        z = z1 + z2 + z3
        
        # Normalize to [0, 1]
        z = (z - z.min()) / (z.max() - z.min())
        
        return z