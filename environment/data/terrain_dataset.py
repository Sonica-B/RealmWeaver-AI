#!/usr/bin/env python
# terrain_dataset.py - Dataset for procedural terrain generation

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

class TerrainDataset(Dataset):
    """Generate procedural terrain heightmaps for training."""
    
    def __init__(self, size=64, num_samples=1000, seed=None):
        """
        Initialize the terrain dataset.
        
        Args:
            size (int): Size of each terrain heightmap (size x size)
            num_samples (int): Number of samples to generate
            seed (int, optional): Random seed for reproducibility
        """
        self.size = size
        self.num_samples = num_samples
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Creating TerrainDataset with {num_samples} samples of size {size}x{size}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a terrain heightmap.
        
        Args:
            idx (int): Index of the sample to generate
            
        Returns:
            torch.Tensor: Flattened heightmap
        """
        # Generate procedural heightmap using different methods based on index
        # to ensure variety in the dataset
        if idx % 4 == 0:
            heightmap = self._generate_perlin_noise()
        elif idx % 4 == 1:
            heightmap = self._generate_fractal_terrain()
        elif idx % 4 == 2:
            heightmap = self._generate_hill_terrain()
        else:
            heightmap = self._generate_mixed_terrain()
        
        # Flatten the heightmap for VAE input
        return torch.from_numpy(heightmap).float().view(-1)
    
    def _generate_perlin_noise(self):
        """Generate terrain using a Perlin noise approximation."""
        # Create grid
        x = np.linspace(0, 5, self.size)
        y = np.linspace(0, 5, self.size)
        xv, yv = np.meshgrid(x, y)
        
        # Generate noise at different frequencies
        noise_layers = []
        
        # Multiple octaves with decreasing amplitude
        for i in range(5):
            freq = 2**i
            amplitude = 1.0 / (2**i)
            phase = np.random.rand() * 2 * np.pi
            
            # Simple sine wave approximation of Perlin noise
            layer = np.sin(xv * freq + phase) * np.cos(yv * freq + phase) * amplitude
            noise_layers.append(layer)
        
        # Combine layers
        heightmap = np.sum(noise_layers, axis=0)
        
        # Normalize to [0, 1]
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        return heightmap
    
    def _generate_fractal_terrain(self):
        """Generate fractal terrain using diamond-square algorithm."""
        # Start with a 2^n + 1 sized array
        size = 2**(int(np.log2(self.size - 1))) + 1
        terrain = np.zeros((size, size))
        
        # Initialize corners
        terrain[0, 0] = np.random.rand()
        terrain[0, size-1] = np.random.rand()
        terrain[size-1, 0] = np.random.rand()
        terrain[size-1, size-1] = np.random.rand()
        
        # Diamond-square algorithm
        step = size - 1
        roughness = 0.5
        
        while step > 1:
            half_step = step // 2
            
            # Diamond step
            for x in range(0, size-1, step):
                for y in range(0, size-1, step):
                    # Average of corners
                    avg = (terrain[x, y] + terrain[x+step, y] + 
                           terrain[x, y+step] + terrain[x+step, y+step]) / 4.0
                    
                    # Add random displacement
                    terrain[x+half_step, y+half_step] = avg + (np.random.rand() - 0.5) * roughness * step
            
            # Square step
            for x in range(0, size-1, half_step):
                for y in range(0, size-1, half_step):
                    if (x % step == 0 and y % step == 0):
                        continue  # Already calculated in diamond step
                    
                    count = 0
                    avg = 0
                    
                    # Add surrounding points if they exist
                    if x >= half_step:
                        avg += terrain[x-half_step, y]
                        count += 1
                    if x + half_step < size:
                        avg += terrain[x+half_step, y]
                        count += 1
                    if y >= half_step:
                        avg += terrain[x, y-half_step]
                        count += 1
                    if y + half_step < size:
                        avg += terrain[x, y+half_step]
                        count += 1
                    
                    # Average and displacement
                    avg /= count
                    terrain[x, y] = avg + (np.random.rand() - 0.5) * roughness * step
            
            # Reduce step size and roughness
            step = half_step
            roughness *= 0.5
        
        # Resize to required size if needed
        if size != self.size:
            from scipy.ndimage import zoom
            terrain = zoom(terrain, self.size/size)
        
        # Normalize to [0, 1]
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        
        return terrain
    
    def _generate_hill_terrain(self):
        """Generate terrain with random hills."""
        # Start with flat terrain
        terrain = np.zeros((self.size, self.size))
        
        # Add random hills
        num_hills = np.random.randint(5, 15)
        
        for _ in range(num_hills):
            # Random hill center
            cx = np.random.randint(0, self.size)
            cy = np.random.randint(0, self.size)
            
            # Random hill properties
            radius = np.random.randint(5, self.size//3)
            height = np.random.rand() * 0.8 + 0.2  # Between 0.2 and 1.0
            
            # Create hill
            for x in range(self.size):
                for y in range(self.size):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        # Add height based on distance from center (bell curve)
                        h = height * np.exp(-(dist**2) / (2 * (radius/2)**2))
                        terrain[y, x] += h
        
        # Normalize to [0, 1]
        if terrain.max() > terrain.min():  # Avoid division by zero
            terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        
        return terrain
    
    def _generate_mixed_terrain(self):
        """Generate terrain mixing different techniques."""
        # Combine different terrain generation methods
        perlin = self._generate_perlin_noise() * 0.6
        hills = self._generate_hill_terrain() * 0.4
        
        # Mix them
        terrain = perlin + hills
        
        # Normalize to [0, 1]
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        
        # Add some random noise for details
        terrain += np.random.rand(self.size, self.size) * 0.05
        terrain = np.clip(terrain, 0, 1)
        
        return terrain

# Test the dataset if run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create dataset
    dataset = TerrainDataset(size=128, num_samples=10)
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(dataset))):
        terrain = dataset[i].view(128, 128).numpy()
        axes[i].imshow(terrain, cmap='terrain')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("terrain_samples.png")
    plt.show()