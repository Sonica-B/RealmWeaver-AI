import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader

class TrainingPipeline:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.setup_logging()
        
    def setup_logging(self):
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(f"{self.config['output_dir']}/training.log"), 
                      logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
    def prepare_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
