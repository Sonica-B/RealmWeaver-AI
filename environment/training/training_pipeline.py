import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging

class TerrainVAETrainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)
    
    def vae_loss(self, recon_x, x, mu, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss
    
    def train(self):
        train_loader = DataLoader(
            self.dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            train_loss = 0
            
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                recon_batch, mu, log_var = self.model(data)
                loss = self.vae_loss(recon_batch, data, mu, log_var)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f'Epoch: {epoch+1}/{self.config["num_epochs"]} '
                                    f'[{batch_idx*len(data)}/{len(train_loader.dataset)} '
                                    f'({100. * batch_idx / len(train_loader):.0f}%)] '
                                    f'Loss: {loss.item()/len(data):.6f}')
            
            avg_loss = train_loss / len(train_loader.dataset)
            self.logger.info(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.6f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch + 1)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
    
    def save_checkpoint(self, epoch):
        checkpoint_dir = Path(self.config['output_dir']) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def generate_terrain(self, num_samples=1, latent_dim=64):
        self.model.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, latent_dim).to(self.device)
            samples = self.model.decode(z)
            
            # Reshape to 2D
            size = int(samples.shape[1] ** 0.5)
            samples = samples.view(num_samples, size, size)
            
            return samples.cpu().numpy()

# Day 1 implementation script
if __name__ == "__main__":
    # Set up environment
    device = setup_environment()
    
    # Create dataset and model
    terrain_size = 64
    input_dim = terrain_size * terrain_size
    model = TerrainVAE(input_dim=input_dim, latent_dim=64, hidden_dim=256)
    dataset = TerrainDataset(size=terrain_size, num_samples=1000)
    
    # Configuration
    config = {
        'output_dir': 'gameworldgen/outputs/terrain_vae',
        'batch_size': 32,
        'num_workers': 2,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'save_interval': 5
    }
    
    # Initialize trainer
    trainer = TerrainVAETrainer(model, dataset, config)
    
    # Train the model
    trainer.train()
    
    # Generate sample terrains
    generated_terrains = trainer.generate_terrain(num_samples=5)
    
    # Visualize terrain (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(generated_terrains), figsize=(15, 3))
        for i, terrain in enumerate(generated_terrains):
            axes[i].imshow(terrain, cmap='terrain')
            axes[i].set_title(f"Sample {i+1}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig("gameworldgen/outputs/terrain_samples.png")
        plt.close()
    except ImportError:
        logging.warning("Matplotlib not installed. Skipping visualization.")