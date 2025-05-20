# generate an image containing several words and colors to test the GPT-4 response.
# for example BLUE in red color, RED in purple etc. 
import sys
from PIL import ImageColor, Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
import math
from loguru import logger
import os
import io
import base64

def color_to_rgb(color: str) -> Tuple[int, int, int]:
    """Convert color name to RGB tuple."""
    logger.debug(f"Converting color {color} to RGB")
    return ImageColor.getrgb(color)

def find_font(font_name: str) -> str:
    """Try to find the font file in common directories, or return the name if it works directly."""
    # Try direct
    try:
        ImageFont.truetype(font_name, 10)
        return font_name
    except Exception:
        pass
    # Try common macOS locations
    common_dirs = [
        "/System/Library/Fonts/Supplemental/",
        "/Library/Fonts/",
        os.path.expanduser("~/Library/Fonts/")
    ]
    for d in common_dirs:
        path = os.path.join(d, font_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Font {font_name} not found in common locations.")

def get_font(font_name: str, font_size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font object with specified name and size."""
    logger.debug(f"Loading font {font_name} with size {font_size} (bold: {bold})")
    if font_name is None:
        logger.info("No font name provided, using default font.")
        return ImageFont.load_default()
    try:
        font_path = find_font(font_name)
        if bold:
            # Try to find a bold variant of the font
            base_name = os.path.splitext(font_path)[0]
            bold_path = f"{base_name} Bold.ttf"
            if os.path.exists(bold_path):
                return ImageFont.truetype(bold_path, font_size)
            # If no bold variant exists, try common bold font names
            bold_fonts = [
                "Arial Bold.ttf",
                "Helvetica Bold.ttf",
                "Times New Roman Bold.ttf",
                "Verdana Bold.ttf"
            ]
            for bold_font in bold_fonts:
                try:
                    return ImageFont.truetype(bold_font, font_size)
                except Exception:
                    continue
            logger.warning("Could not find a bold font variant, using regular font")
        return ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.warning(f"Could not load font {font_name}: {e}, falling back to default font")
        return ImageFont.load_default()

def get_image(width: int, height: int, background_color: str = "white") -> Image.Image:
    """Create a new image with specified dimensions and background color."""
    logger.debug(f"Creating new image with dimensions {width}x{height} and background {background_color}")
    return Image.new("RGB", (width, height), background_color)

class MultipleWords:
    """A class to generate images containing multiple words with different colors.
    
    This class automatically calculates image dimensions and positions words in a grid layout.
    """
    
    def __init__(self, 
                 words: List[str], 
                 colors: List[str], 
                 words_per_row: int = 3,
                 char_spacing: float = 1.5,  # Spacing in units of space character width
                 line_spacing: float = 2.0,  # Spacing in units of character height
                 font_name: str = None,  # Will use default if None
                 font_size: int = 60,
                 background_color: str = "white",
                 alignment: str = "left",  # 'left' or 'center'
                 bold: bool = False):  # Whether to use bold text
        """Initialize the MultipleWords generator."""
        logger.info(f"Initializing MultipleWords with {len(words)} words")
        if len(words) != len(colors):
            logger.error(f"Number of words ({len(words)}) doesn't match number of colors ({len(colors)})")
            raise ValueError("Number of words must match number of colors")
        if alignment not in ("left", "center"):
            raise ValueError("alignment must be 'left' or 'center'")
        
        self.words = words
        self.colors = colors
        self.words_per_row = words_per_row
        self.char_spacing = char_spacing
        self.line_spacing = line_spacing
        self.font_name = font_name
        self.font_size = font_size
        self.background_color = background_color
        self.alignment = alignment
        self.bold = bold
        
        logger.debug(f"Using font size: {self.font_size}")
        # Calculate dimensions and create image
        self._calculate_dimensions()
        self.draw()
    
    def _calculate_dimensions(self):
        """Calculate image dimensions based on content and spacing."""
        logger.info("Calculating image dimensions")
        font = get_font(self.font_name, self.font_size, self.bold)
        space_width = font.getlength(" ")
        max_word_height = 0
        rows = []
        for i in range(0, len(self.words), self.words_per_row):
            row_words = self.words[i:i+self.words_per_row]
            word_widths = [font.getlength(word) for word in row_words]
            row_height = max(font.getbbox(word)[3] - font.getbbox(word)[1] for word in row_words)
            max_word_height = max(max_word_height, row_height)
            # total width: sum of word widths + (n-1) * space_width * char_spacing
            total_row_width = sum(word_widths)
            if len(row_words) > 1:
                total_row_width += (len(row_words) - 1) * space_width * self.char_spacing
            rows.append({
                'words': row_words,
                'word_widths': word_widths,
                'row_width': total_row_width,
                'row_height': row_height
            })
        self.rows = rows
        self.space_width = space_width
        self.max_word_height = max_word_height
        self.v_spacing = self.line_spacing * max_word_height
        # Calculate text block height (all rows + all vertical spacings between rows)
        n_rows = len(rows)
        text_block_height = n_rows * max_word_height + (n_rows - 1) * self.v_spacing
        # Margins
        margin_x = self.space_width
        margin_y = self.space_width  # Use horizontal margin for vertical as well for symmetry
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.text_block_height = text_block_height
        self.width = int(max(row['row_width'] for row in rows) + 2 * margin_x)
        self.height = int(text_block_height + 2 * margin_y)
        logger.info(f"Final image dimensions: {self.width}x{self.height}")
    
    def draw(self):
        """Draw the words on the image in a grid layout."""
        logger.info("Drawing words on image")
        image = get_image(self.width, self.height, self.background_color)
        draw = ImageDraw.Draw(image)
        font = get_font(self.font_name, self.font_size, self.bold)
        # Start y so that the text block is vertically centered
        y = (self.height - self.text_block_height) / 2
        global_word_idx = 0
        for row in self.rows:
            row_words = row['words']
            word_widths = row['word_widths']
            row_width = row['row_width']
            row_height = row['row_height']
            # Alignment
            if self.alignment == "center":
                x = (self.width - row_width) / 2
            else:
                x = self.margin_x
            for idx, word in enumerate(row_words):
                logger.debug(f"Drawing word '{word}' at position ({x}, {y}) with color {self.colors[global_word_idx]}")
                draw.text((x, y), word, fill=self.colors[global_word_idx], font=font)
                x += word_widths[idx]
                if idx < len(row_words) - 1:
                    x += self.space_width * self.char_spacing
                global_word_idx += 1
            y += row_height + self.v_spacing
        self.image = image
        logger.info("Image drawing completed")

    def save(self, path: str):
        """Save the image to a file."""
        logger.info(f"Saving image to {path}")
        try:
            self.image.save(path)
            logger.info("Image saved successfully")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise

    def get_base64(self) -> str:
        """Get the image as a base64-encoded PNG string."""
        logger.debug("Converting image to base64 PNG")
        buffered = io.BytesIO()
        self.image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64
    
    def get_image(self) -> Image.Image:
        """Get the PIL Image object."""
        return self.image

def main():
    """Example usage of MultipleWords class."""
    logger.info("Starting MultipleWords example")
    words = [
        "Red", "Green", "Purple",
        "Brown", "Blue", "Red",
        "Purple", "Red", "Brown",
        "Red", "Green", "Blue"
    ]
    colors = [
        "red", "lime", "darkviolet",
        "peru", "royalblue", "red",
        "lime", "peru", "royalblue",
        "darkviolet", "red", "lime"
    ]
    words_per_row = 20
    char_spacing = 1
    line_spacing = 0.5  # Increase for a bigger gap between rows
    font_size = 30
    font_name = "Arial.ttf"
    background_color = "white"
    alignment = "center"
    bold = True  # Enable bold text
    stimulus = MultipleWords(
        words=words,
        colors=colors,
        words_per_row=words_per_row,
        char_spacing=char_spacing,
        line_spacing=line_spacing,
        font_name=font_name,
        font_size=font_size,
        background_color=background_color,
        alignment=alignment,
        bold=bold
    )
    stimulus.save("output.png")


if __name__ == "__main__":
    main()