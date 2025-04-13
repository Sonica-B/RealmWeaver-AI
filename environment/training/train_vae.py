import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environment.models.vae import VAE
from environment.data.terrain_dataset import SimplifiedTerrainDataset
from shared.utils.training_pipeline import TrainingPipeline

def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

class VAETrainingPipeline(TrainingPipeline):
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.config['output_dir'], f'vae_checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def train(self):
        train_dataloader = self.prepare_dataloader()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        self.logger.info(f"Starting training for {self.config['num_epochs']} epochs")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            train_loss = 0
            
            for batch_idx, data in enumerate(train_dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                recon_batch, mu, log_var = self.model(data)
                loss = vae_loss(recon_batch, data, mu, log_var)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    self.logger.info(f'Epoch: {epoch+1}/{self.config["num_epochs"]} '
                                    f'[{batch_idx*len(data)}/{len(train_dataloader.dataset)} '
                                    f'({100. * batch_idx / len(train_dataloader):.0f}%)] '
                                    f'Loss: {loss.item()/len(data):.6f}')
            
            avg_loss = train_loss / len(train_dataloader.dataset)
            self.logger.info(f'====> Epoch: {epoch+1} Average loss: {avg_loss:.6f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch + 1)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE for terrain generation')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='output directory')
    args = parser.parse_args()
    
    # Model and dataset
    terrain_size = 64
    input_dim = terrain_size * terrain_size
    model = VAE(input_dim=input_dim, latent_dim=64, hidden_dim=256)
    dataset = SimplifiedTerrainDataset(size=terrain_size, num_samples=1000)
    
    # Configuration
    config = {
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_workers': 2,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'save_interval': 5
    }
    
    # Initialize and run training
    pipeline = VAETrainingPipeline(model, dataset, config)
    pipeline.train()
