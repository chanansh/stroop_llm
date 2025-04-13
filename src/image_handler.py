import os
import base64
from typing import Tuple, List, Dict, Any
import random
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from itertools import product
from config import (
    WORDS, COLORS, IMAGE_DIR,
    FONT_NAME, FONT_SIZE, IMG_WIDTH,
    IMG_HEIGHT, BACKGROUND_COLOR
)

class ImageHandler:
    """Handles the creation and management of stimulus images for the Stroop experiment.
    
    This class is responsible for:
    - Generating stimulus images with colored text
    - Managing the image directory
    - Encoding images for API requests
    """
    
    def __init__(self) -> None:
        """Initialize the image handler and create necessary directories."""
        logger.debug("Initializing Image Handler")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        logger.debug(f"Ensuring image directory exists: {IMAGE_DIR}")
        
        # Load font
        try:
            self.font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
            logger.debug(f"Loaded font {FONT_NAME}")
        except Exception as e:
            logger.error(f"Failed to load font {FONT_NAME}: {str(e)}")
            raise
        
        # Cache for images and their base64 encodings
        self.images: Dict[str, Image.Image] = {}
        self.base64_cache: Dict[str, str] = {}
        
        # Generate or load all possible stimulus images
        for word, color in product(WORDS, COLORS):
            path = self._get_image_path(word, color)
            if os.path.exists(path):
                logger.debug(f"Loading existing image: {path}")
                self.images[path] = Image.open(path)
            else:
                logger.debug(f"Generating new image: {path}")
                self.images[path] = self._generate_stimulus(word, color)
                self.images[path].save(path)
            
            # Pre-compute base64 encoding
            with open(path, "rb") as image_file:
                self.base64_cache[path] = base64.b64encode(image_file.read()).decode('utf-8')
        
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
    
    def encode_image(self, image_path: str) -> str:
        """Get the base64 encoding of an image from cache.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Base64 encoded image string from cache
        """
        return self.base64_cache[image_path]
    
    def prepare_image_content(self, image_path: str) -> List[Dict[str, Any]]:
        """Prepare image content for API request.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List[Dict[str, Any]]: List containing the image content in the format
                                required by the API
        """
        try:
            base64_image = self.encode_image(image_path)
            return [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            }]
        except Exception as e:
            logger.error(f"Failed to prepare image content for {image_path}: {str(e)}")
            raise 