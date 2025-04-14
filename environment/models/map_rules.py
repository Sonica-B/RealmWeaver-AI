def create_tilemap_rules():
    """Create rules for 2D tilemap generation"""
    # Define tile types
    tile_types = ['grass', 'water', 'mountain', 'forest', 'road']
    
    # Define rules for each tile type
    rules = {
        'grass': {
            'up': ['grass', 'forest', 'mountain', 'road'],
            'right': ['grass', 'forest', 'road', 'water'],
            'down': ['grass', 'forest', 'road', 'water'],
            'left': ['grass', 'forest', 'mountain', 'road']
        },
        'water': {
            'up': ['water', 'grass'],
            'right': ['water', 'grass'],
            'down': ['water', 'grass'],
            'left': ['water', 'grass']
        },
        'mountain': {
            'up': ['mountain', 'grass'],
            'right': ['mountain', 'grass'],
            'down': ['mountain', 'grass', 'forest'],
            'left': ['mountain', 'grass']
        },
        'forest': {
            'up': ['forest', 'grass', 'mountain'],
            'right': ['forest', 'grass'],
            'down': ['forest', 'grass'],
            'left': ['forest', 'grass']
        },
        'road': {
            'up': ['road', 'grass'],
            'right': ['road', 'grass'],
            'down': ['road', 'grass'],
            'left': ['road', 'grass']
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
            logging.FileHandler("gameworldgen/logs.log"),
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
        wfc.visualize(map_data, output_path="gameworldgen/outputs/generated_map.png")
        
        # Save map data
        import json
        with open("gameworldgen/outputs/map_data.json", "w") as f:
            json.dump(map_data, f)
        
        logging.info("Map generated and saved successfully")
    else:
        logging.error("Map generation failed")