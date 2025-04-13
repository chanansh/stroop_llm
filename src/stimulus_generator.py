import os
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict
from loguru import logger
from itertools import product

from config import (
    WORDS, COLORS, IMAGE_DIR,
    FONT_SIZE, FONT_NAME, IMG_WIDTH,
    IMG_HEIGHT, BACKGROUND_COLOR
)

class StimulusGenerator:
    """Class for generating and managing stimulus images."""
    
    def __init__(self):
        """Initialize the stimulus generator and create all necessary images."""
        logger.debug("Initializing StimulusGenerator")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        
        # Load font
        try:
            self.font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
            logger.debug(f"Loaded font {FONT_NAME}")
        except Exception as e:
            logger.error(f"Failed to load font {FONT_NAME}: {str(e)}")
            raise
        
        # Generate all stimuli
        self.images = {}
        for word in WORDS:
            for color in COLORS:
                path = self.get_stimulus_path(word, color)
                if os.path.exists(path):
                    logger.debug(f"Loading existing image: {path}")
                    self.images[f"{word}_{color}"] = path
                else:
                    logger.debug(f"Generating new image: {path}")
                    self.generate_stimulus(word, color)
        
        logger.debug(f"Initialized with {len(self.images)} images")
    
    def _get_image_path(self, word: str, color: Tuple[int, int, int]) -> str:
        """Generate the path for a stimulus image.
        
        Args:
            word: The word to display
            color: RGB color tuple for the word
            
        Returns:
            Path to the image file
        """
        return os.path.join(IMAGE_DIR, f"{word}_{color[0]}_{color[1]}_{color[2]}.png")
    
    def get_stimulus_path(self, word: str, color: Tuple[int, int, int]) -> str:
        """Get the path to a stimulus image.
        
        Args:
            word: The word to display
            color: RGB color tuple for the word
            
        Returns:
            Path to the image file
        """
        return self._get_image_path(word, color)
    
    def _generate_stimulus(self, word: str, color: Tuple[int, int, int]) -> Image.Image:
        """Generate a stimulus image with the given word and color.
        
        Args:
            word: The word to display
            color: RGB color tuple for the word
            
        Returns:
            The generated PIL Image
        """
        # Create a new image with background color
        image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)
        
        # Get text size and position
        text_bbox = draw.textbbox((0, 0), word, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center the text
        x = (IMG_WIDTH - text_width) // 2
        y = (IMG_HEIGHT - text_height) // 2
        
        # Draw the text
        draw.text((x, y), word, font=self.font, fill=color)
        
        return image 