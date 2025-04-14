#!/usr/bin/env python
# generate_assets.py - Generate game assets for each biome type

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_diffusion_setup import setup_stable_diffusion, generate_asset

def parse_args():
    parser = argparse.ArgumentParser(description="Generate game assets for each biome type")
    parser.add_argument("--map-path", type=str, 
                        default="outputs/generated_maps/map_data.json",
                        help="Path to generated map JSON file")
    parser.add_argument("--output-dir", type=str, 
                        default="outputs/assets",
                        help="Directory to save generated assets")
    parser.add_argument("--style", type=str, default="pixel art", 
                        help="Art style for the generated assets")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA weights (optional)")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generate_assets.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_biome_prompts(style):
    """Get prompts for each biome type"""
    return {
        'water': f"A deep blue water tile for a 2D game, top-down view, {style}",
        'beach': f"A sandy beach tile for a 2D game, top-down view, {style}",
        'plains': f"A grassy plains tile for a 2D game, top-down view, {style}",
        'forest': f"A dense forest tile with trees for a 2D game, top-down view, {style}",
        'mountain': f"A rocky mountain tile for a 2D game, top-down view, {style}",
        'snow': f"A snowy mountain peak tile for a 2D game, top-down view, {style}"
    }

def get_biome_negative_prompts():
    """Get negative prompts for each biome type"""
    # Common negative prompts for all biomes
    common_negatives = "blurry, distorted, ugly, low resolution, text, watermark, signature, bad anatomy"
    
    return {
        'water': f"{common_negatives}, trees, grass, people",
        'beach': f"{common_negatives}, trees, buildings, people",
        'plains': f"{common_negatives}, buildings, mountains, water, people",
        'forest': f"{common_negatives}, buildings, water, people",
        'mountain': f"{common_negatives}, buildings, water, trees, people",
        'snow': f"{common_negatives}, buildings, water, trees, people"
    }

def generate_tileset(pipe, biome_types, style, output_dir):
    """Generate a tileset for the given biome types"""
    logger = setup_logging()
    logger.info(f"Generating tileset with style: {style}")
    
    # Get prompts
    biome_prompts = get_biome_prompts(style)
    biome_negatives = get_biome_negative_prompts()
    
    # Create output directory
    tiles_dir = Path(output_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate assets for each biome type
    tileset = {}
    
    for biome in biome_types:
        if biome not in biome_prompts:
            logger.warning(f"No prompt defined for biome: {biome}, skipping")
            continue
        
        prompt = biome_prompts[biome]
        negative_prompt = biome_negatives.get(biome, "")
        
        # Generate and save asset
        asset_path = tiles_dir / f"{biome}.png"
        image = generate_asset(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=asset_path,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        tileset[biome] = str(asset_path)
    
    # Save tileset metadata
    with open(Path(output_dir) / "tileset.json", "w") as f:
        json.dump(tileset, f, indent=2)
    
    logger.info(f"Generated tileset with {len(tileset)} tiles")
    return tileset

def generate_character_assets(pipe, style, output_dir, num_characters=3):
    """Generate character assets"""
    logger = setup_logging()
    logger.info(f"Generating character assets with style: {style}")
    
    # Character types
    character_types = [
        "warrior",
        "mage",
        "archer",
        "rogue",
        "paladin"
    ]
    
    # Create output directory
    chars_dir = Path(output_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate character assets
    characters = {}
    
    for i in range(min(num_characters, len(character_types))):
        char_type = character_types[i]
        prompt = f"A {char_type} character sprite for a 2D game, full body, facing forward, {style}, high quality, detailed"
        negative_prompt = "blurry, distorted, ugly, low resolution, text, watermark, signature, bad anatomy, missing limbs, extra limbs"
        
        # Generate and save asset
        asset_path = chars_dir / f"{char_type}.png"
        image = generate_asset(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=asset_path,
            num_inference_steps=40,  # More steps for characters
            guidance_scale=8.0  # Higher guidance for better quality
        )
        
        characters[char_type] = str(asset_path)
    
    # Save character metadata
    with open(Path(output_dir) / "characters.json", "w") as f:
        json.dump(characters, f, indent=2)
    
    logger.info(f"Generated {len(characters)} character assets")
    return characters

def generate_object_assets(pipe, style, output_dir):
    """Generate game object assets"""
    logger = setup_logging()
    logger.info(f"Generating object assets with style: {style}")
    
    # Object types
    object_types = {
        "tree": f"A single tree for a 2D game, top-down view, {style}, transparent background",
        "rock": f"A rock/boulder for a 2D game, top-down view, {style}, transparent background",
        "chest": f"A treasure chest for a 2D game, top-down view, {style}, transparent background",
        "house": f"A small house/cottage for a 2D game, top-down view, {style}, transparent background",
        "flower": f"Colorful flowers for a 2D game, top-down view, {style}, transparent background"
    }
    
    # Create output directory
    objects_dir = Path(output_dir) / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate object assets
    objects = {}
    
    for obj_type, prompt in object_types.items():
        negative_prompt = "blurry, distorted, ugly, low resolution, text, watermark, signature"
        
        # Generate and save asset
        asset_path = objects_dir / f"{obj_type}.png"
        image = generate_asset(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=asset_path,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        
        objects[obj_type] = str(asset_path)
    
    # Save object metadata
    with open(Path(output_dir) / "objects.json", "w") as f:
        json.dump(objects, f, indent=2)
    
    logger.info(f"Generated {len(objects)} object assets")
    return objects

def visualize_with_assets(map_data, tileset, output_path):
    """Visualize the map with the generated assets"""
    try:
        from PIL import Image
        import numpy as np
        
        # Get map dimensions
        height = len(map_data)
        width = len(map_data[0])
        
        # Load tile images
        tile_images = {}
        tile_size = 64  # Size to resize tiles to
        
        for biome, path in tileset.items():
            if os.path.exists(path):
                img = Image.open(path)
                tile_images[biome] = img.resize((tile_size, tile_size))
        
        # Create blank image
        result = Image.new('RGBA', (width * tile_size, height * tile_size))
        
        # Place tiles
        for y in range(height):
            for x in range(width):
                biome = map_data[y][x]
                if biome in tile_images:
                    result.paste(tile_images[biome], (x * tile_size, y * tile_size))
        
        # Save result
        result.save(output_path)
        return result
        
    except ImportError:
        logging.warning("PIL not installed. Skipping visualization.")
        return None

def main():
    args = parse_args()
    logger = setup_logging()
    logger.info(f"Generating assets with parameters: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load map data
    try:
        with open(args.map_path, "r") as f:
            map_data = json.load(f)
        logger.info(f"Loaded map data from {args.map_path}")
    except Exception as e:
        logger.error(f"Failed to load map data: {e}")
        logger.info("Using default biome types")
        biome_types = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
    else:
        # Extract unique biome types from map
        biome_types = set()
        for row in map_data:
            biome_types.update(row)
        biome_types = list(biome_types)
        logger.info(f"Found biome types: {biome_types}")
    
    # Set up Stable Diffusion
    pipe = setup_stable_diffusion()
    
    # Load LoRA adapter if specified
    if args.lora_path:
        from stable_diffusion_setup import load_lora_adapter
        pipe = load_lora_adapter(pipe, args.lora_path)
    
    # Generate tileset
    tileset = generate_tileset(pipe, biome_types, args.style, output_dir)
    
    # Generate character assets
    characters = generate_character_assets(pipe, args.style, output_dir)
    
    # Generate object assets
    objects = generate_object_assets(pipe, args.style, output_dir)
    
    # If map data is available, visualize the map with the generated assets
    if 'map_data' in locals():
        logger.info("Visualizing map with generated assets")
        visualize_with_assets(
            map_data,
            tileset,
            output_dir / "visualized_map.png"
        )
    
    logger.info(f"Asset generation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()