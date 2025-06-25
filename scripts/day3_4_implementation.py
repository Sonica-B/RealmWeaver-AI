#!/usr/bin/env python
# day3_4_implementation.py - Run Day 3 and 4 implementations

import os
import subprocess
import sys
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("day3_4_implementation.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(command, logger):
    """Run a command and log the output"""
    logger.info(f"Running command: {command}")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        
        # Stream output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        # Log any errors
        for line in process.stderr:
            logger.error(line.strip())
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def main():
    logger = setup_logging()
    logger.info("Starting Day 3 & 4 implementation")
    
    # Create output directories
    Path("outputs/generated_maps").mkdir(parents=True, exist_ok=True)
    Path("outputs/assets").mkdir(parents=True, exist_ok=True)
    
    # Day 3: Generate map from heightmap using WFC
    logger.info("Running Day 3: Map generation with Wave Function Collapse")
    success = run_command(
        "python -m environment.inference.generate_map " +
        "--heightmap-path outputs/generated_terrain/terrain_1.npy " +
        "--map-width 40 --map-height 40 " +
        "--output-dir outputs/generated_maps",
        logger
    )
    
    if not success:
        logger.error("Map generation failed, but continuing to asset generation")
    
    # Day 4: Generate assets using Stable Diffusion
    logger.info("Running Day 4: Asset generation with Stable Diffusion")
    run_command(
        "python -m environment.inference.generate_assets " +
        "--map-path outputs/generated_maps/map_data.json " +
        "--output-dir outputs/assets " +
        "--style 'pixel art, fantasy game style, top-down 2D'",
        logger
    )
    
    logger.info("Day 3 & 4 implementation complete!")

if __name__ == "__main__":
    main()