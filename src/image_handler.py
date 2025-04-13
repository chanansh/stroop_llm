import os
import base64
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from itertools import product
from .config import ExperimentConfig

class ImageHandler:
    def __init__(self):
        logger.info("Initializing Image Handler")
        self.config = ExperimentConfig
        os.makedirs(self.config.IMAGE_DIR, exist_ok=True)
        logger.debug(f"Ensuring image directory exists: {self.config.IMAGE_DIR}")
        
    def get_image_filename(self, word, color):
        """Generate filename for a stimulus image."""
        filename = f"{self.config.IMAGE_DIR}/{word}_{color[0]}_{color[1]}_{color[2]}.png"
        logger.debug(f"Generated filename: {filename}")
        return filename
    
    def create_stimulus_images(self):
        """Generate stimulus images if they don't already exist."""
        logger.info("Starting stimulus image generation")
        stimuli = list(product(self.config.WORDS, self.config.COLORS))
        logger.debug(f"Total stimuli to generate: {len(stimuli)}")
        
        created_count = 0
        for word, color in stimuli:
            filename = self.get_image_filename(word, color)
            if not os.path.exists(filename):
                logger.debug(f"Creating image for word '{word}' in color {color}")
                self._create_image(word, color, filename)
                created_count += 1
        
        logger.info(f"Image generation complete. Created {created_count} new images.")
    
    def _create_image(self, word, color, filename):
        """Create a single stimulus image."""
        try:
            img = Image.new("RGB", (self.config.IMG_WIDTH, self.config.IMG_HEIGHT), 
                           self.config.BACKGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype(self.config.FONT_NAME, self.config.FONT_SIZE)
                logger.debug(f"Using TrueType font: {self.config.FONT_NAME}")
            except:
                logger.warning(f"Failed to load font {self.config.FONT_NAME}, using default")
                font = ImageFont.load_default()
                
            text_size = draw.textbbox((0, 0), word, font=font)
            text_x = (self.config.IMG_WIDTH - text_size[2]) // 2
            text_y = (self.config.IMG_HEIGHT - text_size[3]) // 2
            
            draw.text((text_x, text_y), word, fill=color, font=font)
            img.save(filename)
            logger.debug(f"Successfully saved image: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to create image {filename}: {str(e)}")
            raise
    
    def encode_image(self, image_path):
        """Encode image to base64."""
        logger.debug(f"Encoding image: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                logger.debug("Image encoded successfully")
                return encoded
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def prepare_image_content(self, image_path):
        """Prepare image content for API request."""
        logger.debug(f"Preparing image content for API: {image_path}")
        base64_image = self.encode_image(image_path)
        return [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": self.config.IMAGE_DETAIL
            }
        }] 