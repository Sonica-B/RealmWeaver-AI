def create_lora_training_command(dataset_path, output_dir, 
                                base_model="runwayml/stable-diffusion-v1-5"):
    """Create command for LoRA fine-tuning"""
    return f"""
    accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \\
        --pretrained_model_name_or_path="{base_model}" \\
        --dataset_name={dataset_path} \\
        --resolution=512 \\
        --train_batch_size=1 \\
        --learning_rate=1e-4 \\
        --max_train_steps=500 \\
        --checkpointing_steps=100 \\
        --validation_prompt="game asset, high quality, detailed" \\
        --output_dir={output_dir} \\
        --lora_r=16 \\
        --lora_alpha=32 \\
        --lora_text_encoder_r=16 \\
        --lora_text_encoder_alpha=32
    """

def load_lora_weights(base_model_pipe, lora_model_path):
    """Load LoRA weights into a base Stable Diffusion pipeline"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import PEFT library for LoRA
        from peft import PeftModel
        import os
        
        # Load the LoRA weights
        base_model_pipe.unet = PeftModel.from_pretrained(
            base_model_pipe.unet, 
            lora_model_path, 
            adapter_name="game_assets"
        )
        
        # If text encoder was also trained
        if any("text_encoder" in n for n in os.listdir(lora_model_path)):
            base_model_pipe.text_encoder = PeftModel.from_pretrained(
                base_model_pipe.text_encoder,
                lora_model_path,
                adapter_name="text_encoder"
            )
        
        logger.info(f"Successfully loaded LoRA weights from {lora_model_path}")
        return base_model_pipe
    
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.info("Continuing with base model")
        return base_model_pipe

# Generate a basic tileset
def generate_game_tileset(pipe, output_dir, style_prompt="pixel art style"):
    """Generate a basic tileset for a 2D game"""
    logger = logging.getLogger(__name__)
    
    tile_types = {
        "grass": "grass terrain tile for 2D game, top-down view",
        "water": "water terrain tile for 2D game, top-down view, blue",
        "mountain": "mountain terrain tile for 2D game, top-down view",
        "forest": "forest terrain tile with trees for 2D game, top-down view",
        "road": "road terrain tile for 2D game, top-down view, dirt path"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    tileset = {}
    for tile_name, tile_desc in tile_types.items():
        prompt = f"{tile_desc}, {style_prompt}, high quality, detailed"
        negative_prompt = "blurry, distorted, low quality"
        
        output_path = os.path.join(output_dir, f"{tile_name}.png")
        
        images = generate_game_asset(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=output_path
        )
        
        tileset[tile_name] = output_path
        logger.info(f"Generated {tile_name} tile")
    
    return tileset

# Day 2 implementation script
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("gameworldgen/logs.log"),
            logging.StreamHandler()
        ]
    )
    
    # Setup Stable Diffusion
    pipe = setup_stable_diffusion()
    
    # Generate a basic tileset
    tileset = generate_game_tileset(
        pipe=pipe,
        output_dir="gameworldgen/outputs/tileset",
        style_prompt="pixel art style, vibrant colors"
    )
    
    # Prepare a sample dataset for LoRA fine-tuning
    # This assumes you have some game assets in a folder
    try:
        prepare_lora_dataset(
            source_folder="gameworldgen/outputs/tileset",
            output_folder="gameworldgen/outputs/lora_dataset",
            caption="game terrain tile pixel art"
        )
        
        # Print LoRA training command
        command = create_lora_training_command(
            dataset_path="gameworldgen/outputs/lora_dataset",
            output_dir="gameworldgen/outputs/lora_model"
        )
        
        print("\nTo train a LoRA model on your assets, run the following command:")
        print(command)
    
    except Exception as e:
        logging.error(f"Error preparing LoRA dataset: {e}")