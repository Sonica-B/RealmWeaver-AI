import os
from PIL import Image
from torch.utils.data import Dataset
import json
import shutil

class GameAssetDataset(Dataset):
    """Dataset for game assets to be used in LoRA fine-tuning"""
    def __init__(self, image_folder, transform=None, caption="game asset"):
        self.image_files = []
        self.captions = []
        self.transform = transform
        
        # Get all image files
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(image_folder, filename))
                self.captions.append(f"{caption}, high quality, detailed")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        caption = self.captions[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption

def prepare_lora_dataset(source_folder, output_folder, caption="game asset"):
    """Prepare dataset for LoRA fine-tuning"""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_folder, exist_ok=True)
    
    metadata = []
    for i, filename in enumerate(os.listdir(source_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(output_folder, f"{i:04d}.png")
            
            # Copy and resize image if needed
            img = Image.open(source_path)
            if max(img.size) > 512:
                img.thumbnail((512, 512))
                img.save(dest_path)
            else:
                shutil.copy(source_path, dest_path)
            
            metadata.append({
                "file_name": f"{i:04d}.png",
                "text": f"{caption}, high quality, detailed"
            })
    
    # Save metadata
    with open(os.path.join(output_folder, "metadata.jsonl"), "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Prepared {len(metadata)} images for LoRA training in {output_folder}")
    return output_folder