# #!/usr/bin/env python
# # generate_assets.py - Generate game assets for each biome type

# import os
# import sys
# import argparse
# import json
# import logging
# from pathlib import Path

# # Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from stable_diffusion_setup import setup_stable_diffusion, generate_asset

# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate game assets for each biome type")
#     parser.add_argument("--map-path", type=str, 
#                         default="outputs/generated_maps/map_data.json",
#                         help="Path to generated map JSON file")
#     parser.add_argument("--output-dir", type=str, 
#                         default="outputs/assets",
#                         help="Directory to save generated assets")
#     parser.add_argument("--style", type=str, default="pixel art", 
#                         help="Art style for the generated assets")
#     parser.add_argument("--lora-path", type=str, default=None,
#                         help="Path to LoRA weights (optional)")
#     return parser.parse_args()

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("generate_assets.log"),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# def get_biome_prompts(style):
#     """Get prompts for each biome type"""
#     return {
#         'water': f"A deep blue water tile for a 2D game, top-down view, {style}",
#         'beach': f"A sandy beach tile for a 2D game, top-down view, {style}",
#         'plains': f"A grassy plains tile for a 2D game, top-down view, {style}",
#         'forest': f"A dense forest tile with trees for a 2D game, top-down view, {style}",
#         'mountain': f"A rocky mountain tile for a 2D game, top-down view, {style}",
#         'snow': f"A snowy mountain peak tile for a 2D game, top-down view, {style}"
#     }

# def get_biome_negative_prompts():
#     """Get negative prompts for each biome type"""
#     # Common negative prompts for all biomes
#     common_negatives = "blurry, distorted, ugly, low resolution, text, watermark, signature, bad anatomy"
    
#     return {
#         'water': f"{common_negatives}, trees, grass, people",
#         'beach': f"{common_negatives}, trees, buildings, people",
#         'plains': f"{common_negatives}, buildings, mountains, water, people",
#         'forest': f"{common_negatives}, buildings, water, people",
#         'mountain': f"{common_negatives}, buildings, water, trees, people",
#         'snow': f"{common_negatives}, buildings, water, trees, people"
#     }

# def generate_tileset(pipe, biome_types, style, output_dir):
#     """Generate a tileset for the given biome types"""
#     logger = setup_logging()
#     logger.info(f"Generating tileset with style: {style}")
    
#     # Get prompts
#     biome_prompts = get_biome_prompts(style)
#     biome_negatives = get_biome_negative_prompts()
    
#     # Create output directory
#     tiles_dir = Path(output_dir) / "tiles"
#     tiles_dir.mkdir(parents=True, exist_ok=True)
    
#     # Generate assets for each biome type
#     tileset = {}
    
#     for biome in biome_types:
#         if biome not in biome_prompts:
#             logger.warning(f"No prompt defined for biome: {biome}, skipping")
#             continue
        
#         prompt = biome_prompts[biome]
#         negative_prompt = biome_negatives.get(biome, "")
        
#         # Generate and save asset
#         asset_path = tiles_dir / f"{biome}.png"
#         image = generate_asset(
#             pipe,
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             output_path=asset_path,
#             num_inference_steps=30,
#             guidance_scale=7.5
#         )
        
#         tileset[biome] = str(asset_path)
    
#     # Save tileset metadata
#     with open(Path(output_dir) / "tileset.json", "w") as f:
#         json.dump(tileset, f, indent=2)
    
#     logger.info(f"Generated tileset with {len(tileset)} tiles")
#     return tileset

# def generate_character_assets(pipe, style, output_dir, num_characters=3):
#     """Generate character assets"""
#     logger = setup_logging()
#     logger.info(f"Generating character assets with style: {style}")
    
#     # Character types
#     character_types = [
#         "warrior",
#         "mage",
#         "archer",
#         "rogue",
#         "paladin"
#     ]
    
#     # Create output directory
#     chars_dir = Path(output_dir) / "characters"
#     chars_dir.mkdir(parents=True, exist_ok=True)
    
#     # Generate character assets
#     characters = {}
    
#     for i in range(min(num_characters, len(character_types))):
#         char_type = character_types[i]
#         prompt = f"A {char_type} character sprite for a 2D game, full body, facing forward, {style}, high quality, detailed"
#         negative_prompt = "blurry, distorted, ugly, low resolution, text, watermark, signature, bad anatomy, missing limbs, extra limbs"
        
#         # Generate and save asset
#         asset_path = chars_dir / f"{char_type}.png"
#         image = generate_asset(
#             pipe,
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             output_path=asset_path,
#             num_inference_steps=40,  # More steps for characters
#             guidance_scale=8.0  # Higher guidance for better quality
#         )
        
#         characters[char_type] = str(asset_path)
    
#     # Save character metadata
#     with open(Path(output_dir) / "characters.json", "w") as f:
#         json.dump(characters, f, indent=2)
    
#     logger.info(f"Generated {len(characters)} character assets")
#     return characters

# def generate_object_assets(pipe, style, output_dir):
#     """Generate game object assets"""
#     logger = setup_logging()
#     logger.info(f"Generating object assets with style: {style}")
    
#     # Object types
#     object_types = {
#         "tree": f"A single tree for a 2D game, top-down view, {style}, transparent background",
#         "rock": f"A rock/boulder for a 2D game, top-down view, {style}, transparent background",
#         "chest": f"A treasure chest for a 2D game, top-down view, {style}, transparent background",
#         "house": f"A small house/cottage for a 2D game, top-down view, {style}, transparent background",
#         "flower": f"Colorful flowers for a 2D game, top-down view, {style}, transparent background"
#     }
    
#     # Create output directory
#     objects_dir = Path(output_dir) / "objects"
#     objects_dir.mkdir(parents=True, exist_ok=True)
    
#     # Generate object assets
#     objects = {}
    
#     for obj_type, prompt in object_types.items():
#         negative_prompt = "blurry, distorted, ugly, low resolution, text, watermark, signature"
        
#         # Generate and save asset
#         asset_path = objects_dir / f"{obj_type}.png"
#         image = generate_asset(
#             pipe,
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             output_path=asset_path,
#             num_inference_steps=30,
#             guidance_scale=7.5
#         )
        
#         objects[obj_type] = str(asset_path)
    
#     # Save object metadata
#     with open(Path(output_dir) / "objects.json", "w") as f:
#         json.dump(objects, f, indent=2)
    
#     logger.info(f"Generated {len(objects)} object assets")
#     return objects

# def visualize_with_assets(map_data, tileset, output_path):
#     """Visualize the map with the generated assets"""
#     try:
#         from PIL import Image
#         import numpy as np
        
#         # Get map dimensions
#         height = len(map_data)
#         width = len(map_data[0])
        
#         # Load tile images
#         tile_images = {}
#         tile_size = 64  # Size to resize tiles to
        
#         for biome, path in tileset.items():
#             if os.path.exists(path):
#                 img = Image.open(path)
#                 tile_images[biome] = img.resize((tile_size, tile_size))
        
#         # Create blank image
#         result = Image.new('RGBA', (width * tile_size, height * tile_size))
        
#         # Place tiles
#         for y in range(height):
#             for x in range(width):
#                 biome = map_data[y][x]
#                 if biome in tile_images:
#                     result.paste(tile_images[biome], (x * tile_size, y * tile_size))
        
#         # Save result
#         result.save(output_path)
#         return result
        
#     except ImportError:
#         logging.warning("PIL not installed. Skipping visualization.")
#         return None

# def main():
#     args = parse_args()
#     logger = setup_logging()
#     logger.info(f"Generating assets with parameters: {args}")
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Load map data
#     try:
#         with open(args.map_path, "r") as f:
#             map_data = json.load(f)
#         logger.info(f"Loaded map data from {args.map_path}")
#     except Exception as e:
#         logger.error(f"Failed to load map data: {e}")
#         logger.info("Using default biome types")
#         biome_types = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
#     else:
#         # Extract unique biome types from map
#         biome_types = set()
#         for row in map_data:
#             biome_types.update(row)
#         biome_types = list(biome_types)
#         logger.info(f"Found biome types: {biome_types}")
    
#     # Set up Stable Diffusion
#     pipe = setup_stable_diffusion()
    
#     # Load LoRA adapter if specified
#     if args.lora_path:
#         from stable_diffusion_setup import load_lora_adapter
#         pipe = load_lora_adapter(pipe, args.lora_path)
    
#     # Generate tileset
#     tileset = generate_tileset(pipe, biome_types, args.style, output_dir)
    
#     # Generate character assets
#     characters = generate_character_assets(pipe, args.style, output_dir)
    
#     # Generate object assets
#     objects = generate_object_assets(pipe, args.style, output_dir)
    
#     # If map data is available, visualize the map with the generated assets
#     if 'map_data' in locals():
#         logger.info("Visualizing map with generated assets")
#         visualize_with_assets(
#             map_data,
#             tileset,
#             output_dir / "visualized_map.png"
#         )
    
#     logger.info(f"Asset generation complete. Results saved to {output_dir}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python
# game_asset_generator.py - Generate large datasets of game assets using Stable Diffusion

#!/usr/bin/env python
# game_asset_generator.py - Generate large datasets of game assets using Stable Diffusion

import os
import sys
import argparse
import logging
import time
import json
import random
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Diffusers imports
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image

# Setup logging
def setup_logging(log_file="asset_generation.log"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Setup Stable Diffusion
def setup_stable_diffusion(model_id="runwayml/stable-diffusion-v1-5", lora_path=None):
    """Set up Stable Diffusion pipeline with optional LoRA weights"""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up Stable Diffusion with model: {model_id}")
    
    # Determine device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model with appropriate settings
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None  # For efficiency
        )
        pipe = pipe.to(device)
        
        # Optimize for inference
        if device == "cuda":
            pipe.enable_attention_slicing()
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                pipe.enable_xformers_memory_efficient_attention()
        
        # Use efficient scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Load LoRA weights if specified
        if lora_path and os.path.exists(lora_path):
            try:
                from peft import PeftModel
                
                logger.info(f"Loading LoRA adapter from {lora_path}")
                # Load LoRA weights
                pipe.unet = PeftModel.from_pretrained(
                    pipe.unet, 
                    lora_path, 
                    adapter_name="game_assets"
                )
                
                # Check if text encoder weights are available
                if os.path.exists(os.path.join(lora_path, "text_encoder")):
                    pipe.text_encoder = PeftModel.from_pretrained(
                        pipe.text_encoder,
                        lora_path,
                        adapter_name="game_assets"
                    )
                
                logger.info("Successfully loaded LoRA adapter")
            except Exception as e:
                logger.error(f"Error loading LoRA weights: {e}")
                logger.info("Continuing with base model")
        
        logger.info(f"Successfully set up Stable Diffusion on {device}")
        return pipe
    
    except Exception as e:
        logger.error(f"Error setting up Stable Diffusion: {e}")
        raise

# Asset generation function
def generate_asset(
    pipe, 
    prompt, 
    negative_prompt="", 
    output_path=None, 
    seed=None,
    num_inference_steps=30, 
    guidance_scale=7.5,
    width=512,
    height=512
):
    """Generate a single image asset"""
    try:
        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(pipe.device).manual_seed(seed)
        
        # Generate image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        
        # Check if images exist and are not empty
        if not hasattr(result, 'images') or len(result.images) == 0:
            logging.error(f"No images generated for prompt: {prompt}")
            return None
            
        image = result.images[0]
        
        # Save image if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
        
        return image
    
    except Exception as e:
        logging.error(f"Error generating asset: {e}")
        return None

# Get prompts for different asset categories
def get_asset_prompts(style):
    """Define prompts for each type of asset"""
    
    # Base style modifiers for consistency
    style_mod = f"{style}, high quality, detailed, game asset, 2D sprite"
    
    # Common negative prompts
    common_neg = "blurry, distorted, ugly, low resolution, text, watermark, signature, bad anatomy, missing limbs, extra limbs, deformed"
    
    asset_prompts = {
        # "characters": {
        #     "archer": f"A 2D sprite of an elven archer with green hood, leather armor, feathered quiver, and yew bow drawn, "
        #     f"forest setting, focused gaze, dynamic pose, high quality, pixel art style, {style_mod}",
        #     "mage": f"A 2D game wizard/mage character sprite with staff and robes, {style_mod}",
        #     "warrior": f"A 2D game warrior character sprite with armor and sword, {style_mod}"
        # },
        # "objects": {
        #     "chest": f"A pixel art, 2D sprite of a fantasy game treasure chest, constructed from dark mahogany with reinforced gold-lined iron bands,lid slightly ajar revealing glittering gold coins, sapphire and ruby gemstones spilling over,intricate carvings of dragons on its sides, faint magical glow emanating from cracks",
        #     "flower": f"A pixel art, 2D sprite of a vibrant fantasy flower cluster, bioluminescent petals ranging from electric blue to neon pink,veins subtly glowing, stems dotted with dew, soft velvet texture on petals,surrounded by small buzzing insects, growing on cracked stone surface, dusk lighting with lens flare",
        #     "house": f"A pixel art, 2D sprite of a detailed fantasy village house, timber-framed cottage with moss-covered stone foundation,thatched roof layered with fallen autumn leaves, ivy climbing up wooden walls,glowing lantern by the door, wooden shutters half-open, plumes of smoke drifting from chimney,sunset sky in background with orange and purple tones, birds perched on rooftop",
        #     "rock": f"A pixel art, 2D sprite of a weathered rock for a 2D RPG, granite surface with visible grain, jagged edges and small cracks,tufts of grass and small fungi growing at base, lichen patch on one side,partial shadow from nearby tree implied by lighting angle, puddle reflecting rock in nearby terrain",
        #     "tree": f"A pixel art, 2D sprite of a massive oak tree sprite, thick gnarled trunk with deep bark fissures, luminous moss creeping up one side,branches twisted dramatically upward, emerald leaves with golden highlights rustling,hidden owl peeking from hollow, ancient runes glowing faintly on lower bark,fireflies floating around roots, night lighting with starlight filtering through canopy"
        # },
        "tiles": {
            "beach": f"A pixel art, 2D sprite of a top-down tile of a sandy tropical beach,tiny seashells and hermit crabs scattered near edge, gentle surf with white foam ripples, palm shadow cutting diagonally across corner",
            "forest": f"A pixel art, 2D sprite of a top-down forest tile, dense canopy shadow, dark green foliage overlapping soft brown dirt paths, twigs and fallen leaves forming natural texture, small mushrooms and red berries peeking through underbrush,hidden bunny or fox partially visible under leaves, foggy morning haze layering the distance",
            "mountain": f"A pixel art, 2D sprite of a rocky mountain tile, craggy gray stone slabs with defined striations,patches of snow melting between rocks, subtle drop shadows indicating elevation",
            "plains": f"A pixel art, 2D sprite of a lush grassy plains tile, tall wheatgrass swaying to the side, scattered wildflowers of red and yellow,, thin game path worn into soil, butterflies mid-flight",
            "snow": f"A pixel art, 2D sprite of a frozen terrain tile, soft powdered snow unevenly distributed, crisp footprints breaking surface,blue-tinted ice reflecting moonlight, twigs and pine needles embedded in frost,visible breath from nearby creature implied, pale sky color palette",
            "water": f"A pixel art, 2D sprite of a deep water tile, layered gradient of navy to turquoise, rippling reflections of a cloudy sky,glints of sunlight sparkling on surface, lily pads drifting slowly, occasional koi shadow underneath, gentle motion lines implying current, top-down RPG style"
        }
    }
    
    # Define negative prompts for each category
    negative_prompts = {
        # "characters": f"{common_neg}, multiple characters, background",
        # "objects": f"{common_neg}, people, text, animals, floating elements",
         "tiles": f"{common_neg}, borders, hard outlines, characters, manmade structures (except house tile)"
    }
    
    return asset_prompts, negative_prompts

# Prompt variation function to create diverse assets
def create_prompt_variation(base_prompt, variation_strength=0.7):
    """Create variations of the base prompt for more diverse assets"""
    
    # Potential modifiers to add to prompts for variation
    color_mods = ["vibrant", "pastel", "muted", "bright", "dark", "colorful", "monochromatic"]
    detail_mods = ["highly detailed", "simple", "minimalist", "ornate", "intricate", "stylized"]
    style_additions = ["cartoon style", "pixel art style", "hand-drawn", "watercolor style", "cell-shaded", "vector art", "flat design", "isometric"]
    lighting_mods = ["dramatic lighting", "soft lighting", "backlit", "rim light", "ambient occlusion"]
    
    # Select random modifiers based on variation strength
    modifiers = []
    categories = [color_mods, detail_mods, style_additions, lighting_mods]
    
    # Higher variation strength means more modifiers
    num_mods = max(1, int(variation_strength * 5))
    
    for _ in range(min(num_mods, len(categories))):
        category = random.choice(categories)
        categories.remove(category)  # Don't use the same category twice
        modifiers.append(random.choice(category))
    
    # Create variation by adding modifiers
    variation = base_prompt
    if modifiers:
        variation = f"{base_prompt}, {', '.join(modifiers)}"
    
    return variation

# Function to handle worker generation safely
def safe_generate_asset(args):
    """Safely generate an asset in a worker thread"""
    try:
        return generate_asset(**args)
    except Exception as e:
        logging.error(f"Worker error generating asset: {e}")
        return None

# Main generation function
def generate_asset_dataset(
    output_dir,
    style="pixel art",
    model_id="runwayml/stable-diffusion-v1-5",
    lora_path=None,
    num_assets_per_type=100,
    batch_size=100,
    start_seed=42,
    variation_strength=0.3
):
    """Generate a large dataset of game assets"""
    logger = setup_logging()
    logger.info(f"Starting asset generation: {num_assets_per_type} assets per type")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup stable diffusion
    pipe = setup_stable_diffusion(model_id, lora_path)
    
    # Get prompts for assets
    asset_prompts, negative_prompts = get_asset_prompts(style)
    
    # Store metadata
    metadata = {
        "model": model_id,
        "lora": lora_path,
        "style": style,
        "generation_date": time.strftime("%Y-%m-%d"),
        "assets": {}
    }
    
    # Generate assets for each category and type
    for category, types in asset_prompts.items():
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Get category-specific negative prompt
        negative_prompt = negative_prompts.get(category, "")
        
        for asset_type, base_prompt in types.items():
            logger.info(f"Generating {num_assets_per_type} {asset_type} assets")
            type_dir = category_dir / asset_type
            type_dir.mkdir(exist_ok=True)
            
            successful_generations = 0
            failed_generations = 0
            
            # Generate assets sequentially
            for i in tqdm(range(num_assets_per_type), desc=f"Generating {asset_type}"):
                # Create variation of the prompt for diversity
                prompt_variation = create_prompt_variation(base_prompt, variation_strength)
                seed = start_seed + i
                output_path = type_dir / f"{asset_type}_{i+1:06d}.png"
                
                try:
                    # Generate directly with the pipeline
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    
                    result = pipe(
                        prompt=prompt_variation,
                        negative_prompt=negative_prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        generator=generator
                    )
                    
                    if hasattr(result, 'images') and len(result.images) > 0:
                        image = result.images[0]
                        os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
                        image.save(str(output_path))
                        successful_generations += 1
                    else:
                        failed_generations += 1
                        logger.error(f"No images generated for prompt: {prompt_variation}")
                        
                except Exception as e:
                    failed_generations += 1
                    logger.error(f"Error generating asset: {e}")
                
                # Add slight pause every 5 generations to avoid memory issues
                if i % 5 == 0 and i > 0:
                    time.sleep(0.5)
            
            # Record metadata for this asset type
            metadata["assets"][f"{category}/{asset_type}"] = {
                "successful": successful_generations,
                "failed": failed_generations,
                "total_attempted": num_assets_per_type,
                "base_prompt": base_prompt,
                "negative_prompt": negative_prompt
            }
    
    # Save metadata
    with open(output_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Asset generation complete. Results saved to {output_dir}")
    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Generate large dataset of game assets")
    parser.add_argument("--output-dir", type=str, default="game_assets_dataset",
                        help="Directory to save generated assets")
    parser.add_argument("--style", type=str, default="pixel art",
                        help="Art style for the generated assets")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA weights (optional)")
    parser.add_argument("--num-assets", type=int, default=100,
                        help="Number of assets to generate per type")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for generation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads")
    parser.add_argument("--variation", type=float, default=0.3,
                        help="Prompt variation strength (0.0 to 1.0)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    generate_asset_dataset(
        output_dir=args.output_dir,
        style=args.style,
        model_id=args.model_id,
        lora_path=args.lora_path,
        num_assets_per_type=args.num_assets,
        batch_size=args.batch_size,
        variation_strength=args.variation
    )