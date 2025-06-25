# #!/usr/bin/env python
# # generate_map.py - Generate game map using heightmap and WFC

# import os
# import sys
# import argparse
# import numpy as np
# import logging
# from pathlib import Path

# # Add project root to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from environment.models.wave_function_collapse import WaveFunctionCollapse
# from environment.models.map_rules import create_tilemap_rules, create_biome_map_from_heightmap

# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate game map using heightmap and WFC")
#     parser.add_argument("--heightmap-path", type=str, 
#                         default="outputs/generated_terrain/terrain_1.npy",
#                         help="Path to heightmap numpy file")
#     parser.add_argument("--map-width", type=int, default=40, 
#                         help="Width of the generated map")
#     parser.add_argument("--map-height", type=int, default=40, 
#                         help="Height of the generated map")
#     parser.add_argument("--output-dir", type=str, 
#                         default="outputs/generated_maps",
#                         help="Directory to save generated maps")
#     return parser.parse_args()

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("generate_map.log"),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# def create_biome_hints_from_heightmap(heightmap, map_width, map_height, tile_types):
#     """
#     Create biome weights based on heightmap for use as hints in WFC
    
#     Args:
#         heightmap: 2D numpy array with height values
#         map_width, map_height: Dimensions of the output map
#         tile_types: List of biome types in order
    
#     Returns:
#         3D array of shape (map_height, map_width, len(tile_types)) with weights
#     """
#     # Resize heightmap to match map dimensions
#     from scipy.ndimage import zoom
#     if heightmap.shape != (map_height, map_width):
#         zoom_factor = (map_height / heightmap.shape[0], map_width / heightmap.shape[1])
#         heightmap = zoom(heightmap, zoom_factor, order=1)
    
#     # Define height ranges for each biome
#     biome_ranges = {
#         'water': (0.0, 0.3),     # 0.0-0.3 height is definitely water
#         'beach': (0.25, 0.4),    # 0.25-0.4 height range for beach
#         'plains': (0.35, 0.6),   # 0.35-0.6 height range for plains
#         'forest': (0.5, 0.75),   # 0.5-0.75 height range for forest
#         'mountain': (0.7, 0.9),  # 0.7-0.9 height range for mountain
#         'snow': (0.85, 1.0)      # 0.85-1.0 height range for snow
#     }
    
#     # Create biome weight array
#     biome_weights = np.zeros((map_height, map_width, len(tile_types)))
    
#     for y in range(map_height):
#         for x in range(map_width):
#             height = heightmap[y, x]
            
#             for i, biome in enumerate(tile_types):
#                 if biome in biome_ranges:
#                     min_h, max_h = biome_ranges[biome]
                    
#                     # Calculate weight based on how close the height is to the range
#                     if min_h <= height <= max_h:
#                         # Within ideal range - high weight
#                         biome_weights[y, x, i] = 10.0
#                     elif height < min_h:
#                         # Below range - decreasing weight
#                         distance = min_h - height
#                         if distance < 0.1:  # Still consider if close
#                             biome_weights[y, x, i] = max(0, 5.0 - distance * 50)
#                     else:  # height > max_h
#                         # Above range - decreasing weight
#                         distance = height - max_h
#                         if distance < 0.1:  # Still consider if close
#                             biome_weights[y, x, i] = max(0, 5.0 - distance * 50)
    
#     return biome_weights

# # Simplify map generation to use heightmap directly
# def create_simple_map_from_heightmap(heightmap, width, height):
#     """Create a simple map directly from heightmap"""
#     from scipy.ndimage import zoom
#     import numpy as np
    
#     # Resize heightmap to match map dimensions
#     if heightmap.shape != (height, width):
#         zoom_factor = (height / heightmap.shape[0], width / heightmap.shape[1])
#         heightmap = zoom(heightmap, zoom_factor, order=1)
    
#     # Define biome thresholds
#     thresholds = {
#         'water': 0.3,    # 0.0-0.3 is water
#         'beach': 0.35,   # 0.3-0.35 is beach
#         'plains': 0.5,   # 0.35-0.5 is plains  
#         'forest': 0.7,   # 0.5-0.7 is forest
#         'mountain': 0.85, # 0.7-0.85 is mountain
#         'snow': 1.0      # 0.85-1.0 is snow
#     }
    
#     # Create map
#     biome_map = []
#     biome_indices = {'water': 0, 'beach': 1, 'plains': 2, 'forest': 3, 'mountain': 4, 'snow': 5}
#     biome_list = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
    
#     for y in range(height):
#         row = []
#         for x in range(width):
#             value = heightmap[y, x]
#             biome = 'snow'  # Default
            
#             # Find the appropriate biome based on height
#             for b, thresh in thresholds.items():
#                 if value <= thresh:
#                     biome = b
#                     break
                    
#             row.append(biome)
#         biome_map.append(row)
    
#     return biome_map

# def main():
#     args = parse_args()
#     logger = setup_logging()
#     logger.info(f"Generating map with parameters: {args}")
    
#     # Create output directory
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Load heightmap
#     try:
#         heightmap = np.load(args.heightmap_path)
#         logger.info(f"Loaded heightmap from {args.heightmap_path}, shape: {heightmap.shape}")
#     except Exception as e:
#         logger.error(f"Failed to load heightmap: {e}")
#         logger.info("Using random heightmap instead")
#         heightmap = np.random.rand(64, 64)
    
#     # Create biome map from heightmap
#     biome_map = create_biome_map_from_heightmap(heightmap)
#     np.save(output_dir / "biome_map.npy", biome_map)
    
#     # Create tile types and rules
#     tile_types, rules = create_tilemap_rules()
    
#     # Create biome hints from heightmap
#     biome_hints = create_biome_hints_from_heightmap(heightmap, args.map_width, args.map_height, tile_types)
    
#     # Initialize Wave Function Collapse
#     # wfc = WaveFunctionCollapse(args.map_width, args.map_height, tile_types, rules)
    
#     # # Generate map using WFC with biome hints
#     # logger.info("Generating map using Wave Function Collapse...")
#     # map_data = wfc.generate(max_retries=5, biome_hints=biome_hints)
    
#     map_data = create_simple_map_from_heightmap(heightmap, args.map_width, args.map_height)

#     if map_data:
#         # Save map data
#         with open(output_dir / "map_data.txt", "w") as f:
#             for row in map_data:
#                 f.write("".join([t[0] for t in row]) + "\n")
        
#         # Save as JSON for easier use in game
#         import json
#         with open(output_dir / "map_data.json", "w") as f:
#             json.dump(map_data, f)
        
#         # Visualize map
#         wfc.visualize(map_data, output_dir / "biome_map.png")
        
#         logger.info(f"Map generated and saved to {output_dir}")
#     else:
#         logger.error("Failed to generate map")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# generate_map.py - Generate game map using heightmap and WFC

import os
import sys
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from environment.models.wave_function_collapse import WaveFunctionCollapse
from environment.models.map_rules import create_tilemap_rules

def parse_args():
    parser = argparse.ArgumentParser(description="Generate game map using heightmap and WFC")
    parser.add_argument("--heightmap-path", type=str, 
                        default="outputs/generated_terrain/terrain_1.npy",
                        help="Path to heightmap numpy file")
    parser.add_argument("--map-width", type=int, default=40, 
                        help="Width of the generated map")
    parser.add_argument("--map-height", type=int, default=40, 
                        help="Height of the generated map")
    parser.add_argument("--output-dir", type=str, 
                        default="outputs/generated_maps",
                        help="Directory to save generated maps")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generate_map.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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

# Simplified map generation to use heightmap directly
def create_simple_map_from_heightmap(heightmap, width, height):
    """Create a simple map directly from heightmap"""
    from scipy.ndimage import zoom
    
    # Resize heightmap to match map dimensions
    if heightmap.shape != (height, width):
        zoom_factor = (height / heightmap.shape[0], width / heightmap.shape[1])
        heightmap = zoom(heightmap, zoom_factor, order=1)
    
    # Define biome thresholds
    thresholds = {
        'water': 0.3,    # 0.0-0.3 is water
        'beach': 0.35,   # 0.3-0.35 is beach
        'plains': 0.5,   # 0.35-0.5 is plains  
        'forest': 0.7,   # 0.5-0.7 is forest
        'mountain': 0.85, # 0.7-0.85 is mountain
        'snow': 1.0      # 0.85-1.0 is snow
    }
    
    # Create map
    biome_map = []
    biome_list = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
    
    for y in range(height):
        row = []
        for x in range(width):
            value = heightmap[y, x]
            biome = 'snow'  # Default
            
            # Find the appropriate biome based on height
            for b, thresh in thresholds.items():
                if value <= thresh:
                    biome = b
                    break
                    
            row.append(biome)
        biome_map.append(row)
    
    return biome_map

def visualize_map(map_data, output_path):
    """Visualize the generated map"""
    # Create color map for different biomes
    biome_colors = {
        'water': 'blue',
        'beach': 'yellow',
        'plains': 'lightgreen',
        'forest': 'darkgreen',
        'mountain': 'brown',
        'snow': 'white'
    }
    
    # Create numerical map
    tile_types = ['water', 'beach', 'plains', 'forest', 'mountain', 'snow']
    tile_to_idx = {tile: i for i, tile in enumerate(tile_types)}
    
    numerical_map = np.array([[tile_to_idx.get(tile, 0) for tile in row] for row in map_data])
    
    # Create custom colormap
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap([biome_colors.get(tile, 'gray') for tile in tile_types])
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(numerical_map, cmap=cmap)
    
    # Add legend
    patches = [plt.Rectangle((0, 0), 1, 1, color=biome_colors.get(tile, 'gray')) for tile in tile_types]
    plt.legend(patches, tile_types, loc='lower right')
    
    plt.title("Generated Biome Map")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return numerical_map

def main():
    args = parse_args()
    logger = setup_logging()
    logger.info(f"Generating map with parameters: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load heightmap
    try:
        heightmap = np.load(args.heightmap_path)
        logger.info(f"Loaded heightmap from {args.heightmap_path}, shape: {heightmap.shape}")
    except Exception as e:
        logger.error(f"Failed to load heightmap: {e}")
        logger.info("Using random heightmap instead")
        heightmap = np.random.rand(64, 64)
    
    # Generate map directly from heightmap
    logger.info("Generating map directly from heightmap...")
    map_data = create_simple_map_from_heightmap(heightmap, args.map_width, args.map_height)

    if map_data:
        # Save map data
        with open(output_dir / "map_data.txt", "w") as f:
            for row in map_data:
                f.write("".join([t[0] for t in row]) + "\n")
        
        # Save as JSON for easier use in game
        import json
        with open(output_dir / "map_data.json", "w") as f:
            json.dump(map_data, f)
        
        # Visualize map
        visualize_map(map_data, output_dir / "biome_map.png")
        
        logger.info(f"Map generated and saved to {output_dir}")
    else:
        logger.error("Failed to generate map")

if __name__ == "__main__":
    main()