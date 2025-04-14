import os
import base64
from typing import Tuple, Dict
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from itertools import product
from config import (
    WORDS, COLORS, IMAGE_DIR,
    FONT_NAME, FONT_SIZE, IMG_WIDTH,
    IMG_HEIGHT, BACKGROUND_COLOR
)
import math

class ImageHandler:
    """Handles the creation, management, and encoding of stimulus images for the Stroop experiment.
    
    This class is responsible for:
    - Generating stimulus images with colored text
    - Managing the image directory
    - Caching images and their base64 encodings
    - Encoding images for API requests
    """
    
    def __init__(self, overwrite_existing: bool = False) -> None:
        """Initialize the image handler and create necessary directories.
        
        Args:
            overwrite_existing: If True, existing stimulus images will be regenerated
        """
        logger.debug("Initializing Image Handler")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        logger.debug(f"Ensuring image directory exists: {IMAGE_DIR}")
        
        # Clean up existing images if overwriting
        if overwrite_existing:
            logger.debug("Cleaning up existing images")
            for file in os.listdir(IMAGE_DIR):
                if file.endswith('.png'):
                    os.remove(os.path.join(IMAGE_DIR, file))
        
        # Load font with bold weight
        try:
            self.font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
            # Create a bold version of the font
            self.bold_font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
            logger.debug(f"Loaded font {FONT_NAME} at size {FONT_SIZE}")
        except Exception as e:
            logger.error(f"Failed to load font {FONT_NAME}: {str(e)}")
            raise
        
        # Cache for images and their base64 encodings
        self.images: Dict[str, Image.Image] = {}
        self.base64_cache: Dict[Tuple[str, Tuple[int, int, int]], str] = {}
        self.path_cache: Dict[Tuple[str, Tuple[int, int, int]], str] = {}
        
        # Initialize all stimuli
        self._initialize_stimuli(overwrite_existing)
        
        logger.debug(f"Initialized with {len(self.images)} images")
        
    def _initialize_stimuli(self, overwrite_existing: bool) -> None:
        """Initialize all stimulus images and their base64 encodings.
        
        Args:
            overwrite_existing: If True, existing stimulus images will be regenerated
        """
        for word, color in product(WORDS, COLORS):
            path = self._get_image_path(word, color)
            self.path_cache[(word, color)] = path
            
            if os.path.exists(path) and not overwrite_existing:
                logger.debug(f"Loading existing image: {path}")
                self.images[path] = Image.open(path)
            else:
                logger.debug(f"Generating new image: {path}")
                self.images[path] = self._generate_stimulus(word, color)
                self.images[path].save(path)
            
            # Pre-compute base64 encoding
            with open(path, "rb") as image_file:
                self.base64_cache[(word, color)] = base64.b64encode(image_file.read()).decode('utf-8')
        
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
        return self.path_cache.get((word, color), self._get_image_path(word, color))
    
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
        text_bbox = draw.textbbox((0, 0), word, font=self.bold_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center the text
        x = (IMG_WIDTH - text_width) // 2
        y = (IMG_HEIGHT - text_height) // 2
        
        # Draw the text with bold font
        draw.text((x, y), word, font=self.bold_font, fill=color)
        
        return image
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def get_stimuli_encoded(self, word: str, color: Tuple[int, int, int]) -> str:
        """Get the base64 encoding of an image from cache.
        
        Args:
            word: The word to display
            color: RGB color tuple for the word
            
        Returns:
            str: Base64 encoded image string from cache
        """
        return self.base64_cache[(word, color)] 
    

def main():
    image_handler = ImageHandler(overwrite_existing=True)

if __name__ == "__main__":
    main()
