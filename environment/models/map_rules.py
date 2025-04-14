import numpy as np
import logging
from pathlib import Path

def create_biome_map_from_heightmap(heightmap, thresholds=None):
    """
    Convert a heightmap to a biome map based on height thresholds.
    
    Args:
        heightmap: 2D numpy array with height values (0.0-1.0)
        thresholds: Dictionary of biome thresholds. If None, uses default thresholds.
    
    Returns:
        2D numpy array with biome indices
    """
    if thresholds is None:
        thresholds = {
            'water': 0.3,       # Below this is water
            'beach': 0.35,      # Below this is beach
            'plains': 0.5,      # Below this is plains
            'forest': 0.7,      # Below this is forest
            'mountain': 0.85,   # Below this is mountain
            'snow': 1.0         # Below this is snow
        }
    
    biome_indices = {
        'water': 0, 
        'beach': 1, 
        'plains': 2, 
        'forest': 3, 
        'mountain': 4, 
        'snow': 5
    }
    
    biome_map = np.zeros(heightmap.shape, dtype=np.int32)
    
    # Assign biomes based on height thresholds
    for biome, threshold in reversed(list(thresholds.items())):
        biome_map[heightmap <= threshold] = biome_indices[biome]
    
    return biome_map

def create_tilemap_rules():
    """Create more flexible rules for 2D tilemap generation"""
    # Define tile types
    tile_types = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
    
    # Define rules with more flexibility between adjacent biomes
    rules = {
        'water': {
            'up': ['water', 'beach'], 
            'right': ['water', 'beach'],
            'down': ['water', 'beach'],
            'left': ['water', 'beach']
        },
        'beach': {
            'up': ['beach', 'plains', 'water'],
            'right': ['beach', 'plains', 'water'],
            'down': ['beach', 'plains', 'water'],
            'left': ['beach', 'plains', 'water']
        },
        'plains': {
            'up': ['plains', 'forest', 'beach', 'mountain'],
            'right': ['plains', 'forest', 'beach', 'mountain'],
            'down': ['plains', 'forest', 'beach', 'mountain'],
            'left': ['plains', 'forest', 'beach', 'mountain']
        },
        'forest': {
            'up': ['forest', 'mountain', 'plains', 'snow'],
            'right': ['forest', 'mountain', 'plains', 'snow'],
            'down': ['forest', 'mountain', 'plains', 'snow'],
            'left': ['forest', 'mountain', 'plains', 'snow']
        },
        'mountain': {
            'up': ['mountain', 'snow', 'forest', 'plains'],
            'right': ['mountain', 'snow', 'forest', 'plains'],
            'down': ['mountain', 'snow', 'forest', 'plains'],
            'left': ['mountain', 'snow', 'forest', 'plains']
        },
        'snow': {
            'up': ['snow', 'mountain', 'forest'],
            'right': ['snow', 'mountain', 'forest'],
            'down': ['snow', 'mountain', 'forest'],
            'left': ['snow', 'mountain', 'forest']
        }
    }
    
    return tile_types, rules

# Day 3 implementation script
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create tiles and rules
    tile_types, rules = create_tilemap_rules()
    
    # Initialize WFC
    wfc = WaveFunctionCollapse(width=20, height=20, tile_types=tile_types, rules=rules)
    
    # Generate map
    map_data = wfc.generate()
    
    if map_data:
        # Visualize map
        wfc.visualize(map_data, output_path="outputs/generated_map.png")
        
        # Save map data
        import json
        with open("outputs/map_data.json", "w") as f:
            json.dump(map_data, f)
        
        logging.info("Map generated and saved successfully")
    else:
        logging.error("Map generation failed")