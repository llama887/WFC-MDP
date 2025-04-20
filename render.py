import pygame
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List, Any

class WFCRenderer:
    """
    Pygame renderer for Wave Function Collapse environments.
    Visualizes both collapsed and uncollapsed cells.
    """
    
    def __init__(self, 
                 json_path: str, 
                 screen_width: int = 800, 
                 screen_height: int = 600,
                 tile_size: int = 32,
                 background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize the WFC renderer.
        
        Args:
            json_path: Path to the JSON file containing tile information
            screen_width: Width of the Pygame window
            screen_height: Height of the Pygame window
            tile_size: Size of each tile in pixels
            background_color: RGB color for the background
        """
        # Initialize Pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
            
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.tile_size = tile_size
        self.background_color = background_color
        
        # Load tile data
        self.tile_data = self._load_tile_data(json_path)
        
        # Create dictionary to map from tile ID to name
        self.id_to_name = {tile_info["id"]: tile_name for tile_name, tile_info in self.tile_data.items()}
        
        # Initialize screen
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Wave Function Collapse")
        
        # Load tile images
        self.tile_images = self._load_tile_images()
        
        # Font for rendering text
        self.font = pygame.font.SysFont(None, 20)
    
    def _load_tile_data(self, json_path: str) -> Dict[str, Any]:
        """Load tile data from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data["tiles"]
    
    def _load_tile_images(self) -> Dict[int, pygame.Surface]:
        """Load all tile images and return a dictionary mapping tile IDs to their images"""
        tile_images = {}
        
        for tile_name, tile_data in self.tile_data.items():
            tile_id = tile_data["id"]
            image_path = tile_data["image"]
            
            # Ensure path is using correct directory separators for the OS
            normalized_path = os.path.normpath(image_path)
            try:
                # Try to load the image
                image = pygame.image.load(normalized_path)
                
                # Scale if necessary
                if image.get_width() != self.tile_size or image.get_height() != self.tile_size:
                    image = pygame.transform.scale(image, (self.tile_size, self.tile_size))
                
                tile_images[tile_id] = image
            except pygame.error:
                # If image loading fails, create a placeholder
                placeholder = self._create_placeholder(tile_name, tile_id)
                tile_images[tile_id] = placeholder
                print(f"Failed to load image: {normalized_path}")
        
        return tile_images
    
    def _create_placeholder(self, tile_name: str, tile_id: int) -> pygame.Surface:
        """Create a placeholder image for tiles with missing images"""
        surf = pygame.Surface((self.tile_size, self.tile_size))
        
        # Fill with a light gray color
        surf.fill((200, 200, 200))
        
        # Add tile name as text
        text = self.font.render(f"{tile_name} ({tile_id})", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.tile_size // 2, self.tile_size // 2))
        surf.blit(text, text_rect)
        
        # Add a border
        pygame.draw.rect(surf, (100, 100, 100), (0, 0, self.tile_size, self.tile_size), 1)
        
        return surf
    
    def render(self, wave_state: np.ndarray, show_entropy: bool = False) -> pygame.Surface:
        """
        Render the current wave state.
        
        Args:
            wave_state: 3D numpy array of shape (height, width, num_patterns)
                        where each value is True/False indicating if a pattern is possible
            show_entropy: Whether to show entropy values for uncollapsed cells
            
        Returns:
            The Pygame surface that was rendered to
        """
        height, width, num_patterns = wave_state.shape
        
        # Calculate grid dimensions
        grid_width = width * self.tile_size
        grid_height = height * self.tile_size
        
        # Adjust screen size if needed
        if grid_width > self.screen_width or grid_height > self.screen_height:
            self.screen_width = max(grid_width, self.screen_width)
            self.screen_height = max(grid_height, self.screen_height)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        # Fill background
        self.screen.fill(self.background_color)
        
        # Render each cell
        for y in range(height):
            for x in range(width):
                cell_state = wave_state[y, x]
                
                # Count number of possible patterns
                possible_patterns = np.where(cell_state)[0]
                num_possible = len(possible_patterns)
                
                if num_possible == 1:
                    # Cell is collapsed - render the corresponding tile
                    pattern_id = possible_patterns[0]
                    if pattern_id in self.tile_images:
                        self.screen.blit(self.tile_images[pattern_id], (x * self.tile_size, y * self.tile_size))
                    else:
                        # Missing image - render a red square
                        pygame.draw.rect(
                            self.screen,
                            (255, 0, 0),
                            (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size),
                        )
                else:
                    # Cell is not collapsed - render a white square with optional entropy info
                    cell_color = (255, 255, 255)  # White
                    
                    # Draw the cell
                    pygame.draw.rect(
                        self.screen,
                        cell_color,
                        (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size),
                    )
                    
                    # Add border
                    pygame.draw.rect(
                        self.screen,
                        (200, 200, 200),
                        (x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size),
                        1
                    )
                    
                    if show_entropy:
                        # Show number of possible patterns
                        text = self.font.render(str(num_possible), True, (0, 0, 0))
                        text_rect = text.get_rect(center=(
                            x * self.tile_size + self.tile_size // 2,
                            y * self.tile_size + self.tile_size // 2
                        ))
                        self.screen.blit(text, text_rect)
        
        # Update display
        pygame.display.flip()
        
        return self.screen
    
    def render_wfc_env(self, wfc_env, show_entropy: bool = False) -> pygame.Surface:
        """
        Render a WFCEnv instance.
        
        Args:
            wfc_env: The WFCEnv instance to render
            show_entropy: Whether to show entropy values for uncollapsed cells
            
        Returns:
            The Pygame surface that was rendered to
        """
        wave_state = wfc_env.get_obs()
        return self.render(wave_state, show_entropy)
    
    def handle_events(self) -> bool:
        """
        Handle Pygame events.
        
        Returns:
            bool: True if the window should close, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False
    
    def close(self):
        """Close the renderer"""
        pygame.quit()


# Helper function to initialize renderer from a WFCEnv instance
def create_renderer_for_env(wfc_env, json_path, **kwargs):
    """
    Create a renderer for a WFCEnv instance.
    
    Args:
        wfc_env: The WFCEnv instance
        json_path: Path to the JSON file with tile information
        **kwargs: Additional arguments to pass to the WFCRenderer constructor
        
    Returns:
        WFCRenderer: The renderer instance
    """
    renderer = WFCRenderer(json_path, **kwargs)
    return renderer