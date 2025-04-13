import os
import math
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trial configuration
PRACTICE_TRIALS: int = 10
STIMULI_COUNT: int = 360  # 6 blocks * 60 trials
MESSAGES_PER_STIMULUS: int = 2  # text stimulus + image stimulus
MESSAGES_FOR_FEEDBACK: int = 2  # feedback + feedback response
MESSAGES_FOR_RESPONSE: int = 1  # user response
MESSAGES_PER_PRACTICE_TRIAL: int = MESSAGES_PER_STIMULUS + MESSAGES_FOR_FEEDBACK + MESSAGES_FOR_RESPONSE

# Stimulus configuration
WORDS: List[str] = ["BLUE", "RED", "XXXX"]
COLORS: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
BACKGROUND_COLOR: Tuple[int, int, int] = (192, 192, 192)  # Silver
FONT_NAME: str = "Courier New"
IMG_WIDTH: int = 800
IMG_HEIGHT: int = 400

# Key mappings for responses
KEY_MAPPING: Dict[Tuple[int, int, int], str] = {
    (0, 0, 255): "b",  # Blue color -> 'b'
    (255, 0, 0): "m"  # Red color -> 'm'
}

# Display configuration
VIEWING_DISTANCE_CM: float = 63.5  # Viewing distance in cm
DPI: int = 96  # Approximate screen DPI
CM_TO_PIXELS: float = DPI / 2.54  # Conversion factor
HEIGHT_ANGLE: float = (2.28 + 3.16) / 2  # Average height visual angle
WIDTH_ANGLE: float = (10.44 + 10.8) / 2  # Average width visual angle
FONT_SIZE: int = int(2 * VIEWING_DISTANCE_CM * math.tan(math.radians(HEIGHT_ANGLE / 2)) * CM_TO_PIXELS)

# File paths
IMAGE_DIR: str = "stimuli_images"
LOG_FILE: str = "experiment.log"
RESULTS_DIR: str = "results"

# Experiment settings
DRY_RUN: bool = False  # Set to True for testing without OpenAI API
NUM_PARTICIPANTS: int = 3
SYSTEM_MESSAGE: str = """You are participating in a Stroop experiment. Your task is to identify the color of the text, not the word itself.

Rules:
1. Respond with 'b' if the text color is blue
2. Respond with 'm' if the text color is red
3. Ignore the meaning of the word, focus only on its color
4. Respond as quickly and accurately as possible

During practice trials, you will receive feedback on your responses. Use this feedback to improve your performance."""

# API configuration
MODEL: str = "gpt-4o"  # Vision model for image analysis
MAX_RETRIES: int = 3
RETRY_DELAY: int = 1  # seconds

# API Key
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables") 