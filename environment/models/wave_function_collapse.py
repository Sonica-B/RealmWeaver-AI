import random
import numpy as np
from collections import deque
import logging

class WaveFunctionCollapse:
    def __init__(self, width, height, tile_types, rules, constraints=None):
        """
        Initialize the WFC algorithm
        
        Args:
            width, height: Dimensions of the map
            tile_types: List of possible tile types
            rules: Dictionary mapping each tile type to allowed neighbor tiles in each direction
            constraints: Optional 2D array with initial constraints (None for any tile type)
        """
        self.width = width
        self.height = height
        self.tile_types = tile_types
        self.rules = rules
        self.logger = logging.getLogger(__name__)
        
        # Initialize grid with all possibilities, or apply constraints
        self.grid = []
        self.collapsed = []
        
        for y in range(height):
            grid_row = []
            collapsed_row = []
            
            for x in range(width):
                if constraints is not None and constraints[y][x] is not None:
                    # Apply constraint if specified
                    allowed_tile = constraints[y][x]
                    if isinstance(allowed_tile, str):
                        # If string, find the index
                        if allowed_tile in tile_types:
                            grid_row.append([allowed_tile])
                        else:
                            self.logger.warning(f"Unknown tile type '{allowed_tile}' at ({x}, {y}), using all types")
                            grid_row.append(list(tile_types))
                    elif isinstance(allowed_tile, int):
                        # If integer, use as index
                        if 0 <= allowed_tile < len(tile_types):
                            grid_row.append([tile_types[allowed_tile]])
                        else:
                            self.logger.warning(f"Invalid tile index {allowed_tile} at ({x}, {y}), using all types")
                            grid_row.append(list(tile_types))
                    else:
                        grid_row.append(list(tile_types))
                else:
                    grid_row.append(list(tile_types))
                
                collapsed_row.append(len(grid_row[-1]) == 1)
            
            self.grid.append(grid_row)
            self.collapsed.append(collapsed_row)
    
    def get_entropy(self, x, y):
        """Calculate entropy (number of possible states) for a cell"""
        if self.collapsed[y][x]:
            return 0
        return len(self.grid[y][x])
    
    def find_min_entropy_cell(self):
        """Find cell with minimum entropy"""
        min_entropy = float('inf')
        min_cells = []
        
        for y in range(self.height):
            for x in range(self.width):
                if not self.collapsed[y][x]:
                    entropy = self.get_entropy(x, y)
                    if entropy < min_entropy and entropy > 1:
                        min_entropy = entropy
                        min_cells = [(x, y)]
                    elif entropy == min_entropy:
                        min_cells.append((x, y))
        
        if not min_cells:
            return None
        
        return random.choice(min_cells)
    
    def collapse_cell(self, x, y):
        """Collapse a cell to a single state"""
        possible_tiles = self.grid[y][x]
        # Pick a random tile weighted by frequency if available
        chosen_tile = random.choice(possible_tiles)
        
        # Set cell to only chosen tile
        self.grid[y][x] = [chosen_tile]
        self.collapsed[y][x] = True
        
        self.logger.debug(f"Collapsed cell ({x}, {y}) to {chosen_tile}")
        return chosen_tile
    
    def propagate(self, start_x, start_y):
        """Propagate constraints from a collapsed cell"""
        stack = deque([(start_x, start_y)])
        
        while stack:
            x, y = stack.popleft()
            
            # Skip if cell has no valid options (contradiction)
            if not self.grid[y][x]:
                self.logger.debug(f"Contradiction at ({x}, {y})")
                return False
            
            # Check neighbors
            directions = [('up', 0, -1), ('right', 1, 0), ('down', 0, 1), ('left', -1, 0)]
            
            for direction, dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Skip if out of bounds
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                
                # Skip if already collapsed with single option
                if self.collapsed[ny][nx] and len(self.grid[ny][nx]) == 1:
                    continue
                
                # Get allowed neighbors for each possible tile in current cell
                current_possible_tiles = self.grid[y][x]
                allowed_neighbors = set()
                
                for tile in current_possible_tiles:
                    # Get the opposite direction for rules
                    opposite_dir = {'up': 'down', 'right': 'left', 'down': 'up', 'left': 'right'}[direction]
                    
                    # Add tiles that can be neighbors in this direction
                    for neighbor_tile in self.tile_types:
                        # Check if current tile allows this neighbor in this direction
                        if neighbor_tile in self.rules[tile][direction]:
                            # Also check if neighbor allows current tile in opposite direction
                            if any(tile in self.rules[neighbor_tile][opposite_dir] for tile in current_possible_tiles):
                                allowed_neighbors.add(neighbor_tile)
                
                # Update possible tiles in neighbor cell
                before_update = len(self.grid[ny][nx])
                self.grid[ny][nx] = [t for t in self.grid[ny][nx] if t in allowed_neighbors]
                
                # If possibilities changed, add neighbor to stack
                if len(self.grid[ny][nx]) < before_update:
                    self.logger.debug(f"Updated cell ({nx}, {ny}) possibilities: {before_update} -> {len(self.grid[ny][nx])}")
                    stack.append((nx, ny))
                
                # Check for contradictions
                if len(self.grid[ny][nx]) == 0:
                    self.logger.debug(f"Contradiction at ({nx}, {ny})")
                    return False
        
        return True
    
    def generate(self, max_retries=5, biome_hints=None):
        """
        Generate a complete map
        
        Args:
            max_retries: Maximum number of retry attempts
            biome_hints: Optional 2D array with preferred biome weights
                         Higher values increase chance of selecting that biome
        
        Returns:
            2D array of tile types or None if generation fails
        """
        for attempt in range(max_retries):
            self.logger.info(f"Attempt {attempt+1} to generate map")
            
            # Reset grid
            self.grid = [[list(self.tile_types) for _ in range(self.width)] for _ in range(self.height)]
            self.collapsed = [[False for _ in range(self.width)] for _ in range(self.height)]
            
            success = True
            
            while True:
                # Find cell with minimum entropy
                min_entropy_cell = self.find_min_entropy_cell()
                
                # If all cells are collapsed, we're done
                if min_entropy_cell is None:
                    if all(all(self.collapsed[y][x] for x in range(self.width)) for y in range(self.height)):
                        self.logger.info("Map generation successful")
                        break
                    else:
                        # We have a contradiction
                        self.logger.warning("Contradiction in map generation")
                        success = False
                        break
                
                # Collapse cell and propagate
                x, y = min_entropy_cell
                
                # Apply biome hints if available
                if biome_hints is not None:
                    possible_tiles = self.grid[y][x]
                    weights = []
                    
                    for tile in possible_tiles:
                        tile_idx = self.tile_types.index(tile)
                        # Apply hint weight if available, otherwise use 1.0
                        weight = biome_hints[y][x][tile_idx] if biome_hints[y][x] is not None else 1.0
                        weights.append(weight)
                    
                    # Choose tile weighted by biome hints
                    if sum(weights) > 0:
                        chosen_idx = random.choices(range(len(possible_tiles)), weights=weights)[0]
                        self.grid[y][x] = [possible_tiles[chosen_idx]]
                        self.collapsed[y][x] = True
                    else:
                        self.collapse_cell(x, y)
                else:
                    self.collapse_cell(x, y)
                
                propagation_success = self.propagate(x, y)
                
                if not propagation_success:
                    # We have a contradiction
                    self.logger.warning("Contradiction during propagation")
                    success = False
                    break
            
            if success:
                # Convert to final map (taking the first/only option for each cell)
                result = [[self.grid[y][x][0] for x in range(self.width)] for y in range(self.height)]
                return result
        
        self.logger.error(f"Failed to generate map after {max_retries} attempts")
        return None
    
    def visualize(self, map_data=None, output_path=None):
        """Visualize the generated map (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            if map_data is None:
                # Use current state - take first option or empty if none
                map_data = []
                for y in range(self.height):
                    row = []
                    for x in range(self.width):
                        if self.grid[y][x]:
                            row.append(self.grid[y][x][0])
                        else:
                            row.append(None)
                    map_data.append(row)
            
            # Create color map for different biomes
            biome_colors = {
                'water': 'blue',
                'beach': 'yellow',
                'plains': 'lightgreen',
                'forest': 'darkgreen',
                'mountain': 'brown',
                'snow': 'white'
            }
            
            # Create custom colormap
            cmap = mcolors.ListedColormap([biome_colors.get(tile, 'gray') for tile in self.tile_types])
            
            # Convert map_data to numerical indices
            tile_to_idx = {tile: i for i, tile in enumerate(self.tile_types)}
            numerical_map = np.array([[tile_to_idx[tile] for tile in row] for row in map_data])
            
            # Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(numerical_map, cmap=cmap)
            
            # Add legend
            patches = [plt.Rectangle((0, 0), 1, 1, color=biome_colors.get(tile, 'gray')) for tile in self.tile_types]
            plt.legend(patches, self.tile_types, loc='lower right')
            
            plt.title("Generated Biome Map")
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
                self.logger.info(f"Map visualization saved to {output_path}")
            
            plt.close()
            
            return numerical_map
            
        except ImportError:
            self.logger.warning("Matplotlib not installed. Skipping visualization.")
            return None