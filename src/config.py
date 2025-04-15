import os
import math
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trial configuration
PRACTICE_TRIALS: int = 0
TEST_TRIALS: int = 360  # 6 blocks * 60 trials
MESSAGES_PER_SYSTEM: int = 1  # system message
MESSAGES_PER_STIMULUS: int = 1  # stimulus message + AI response
MESSAGES_FOR_FEEDBACK: int = 2  # feedback + feedback response
MESSAGES_FOR_RESPONSE: int = 1  # user response
MESSAGES_PER_PRACTICE_TRIAL: int = MESSAGES_PER_STIMULUS + MESSAGES_FOR_FEEDBACK + MESSAGES_FOR_RESPONSE

# Display configuration
VIEWING_DISTANCE_CM: float = 63.5  # Viewing distance in cm

# Fixed image dimensions
IMG_WIDTH: int = 400
IMG_HEIGHT: int = 200
FONT_SIZE: int = int(IMG_HEIGHT * 0.2)  # Use 80% of image height for font

# Stimulus configuration
WORDS: List[str] = ["BLUE", "RED", "XXXX"]
COLORS: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
BACKGROUND_COLOR: Tuple[int, int, int] = (192, 192, 192)  # Silver
FONT_NAME: str = "Courier New"

# Key mappings for responses
RESPONSE_BUTTON_BLUE: str = "b"  # Button for blue color response
RESPONSE_BUTTON_RED: str = "m"   # Button for red color response

KEY_MAPPING: Dict[Tuple[int, int, int], str] = {
    (0, 0, 255): RESPONSE_BUTTON_BLUE,  # Blue color -> 'b'
    (255, 0, 0): RESPONSE_BUTTON_RED  # Red color -> 'm'
}

# File paths
IMAGE_DIR: str = "stimuli_images"
LOG_FILE: str = "experiment.log"
RESULTS_DIR: str = "results"

# Experiment settings
DRY_RUN: bool = False  # Set to True for testing without OpenAI API
NUM_PARTICIPANTS: int = 30
SYSTEM_MESSAGE: str = f"""You are participating in a Stroop experiment. Your task is to identify the color of the text, not the word itself.

Rules:
1. Respond with '{RESPONSE_BUTTON_BLUE}' if the text color is blue
2. Respond with '{RESPONSE_BUTTON_RED}' if the text color is red
3. Ignore the meaning of the word, focus only on its color
4. Respond as quickly and accurately as possible

During practice trials, you will receive feedback on your responses. Use this feedback to improve your performance."""

# API configuration
MODEL: str = "gpt-4o-mini"  # DO NOT CHANGE THIS
MAX_RETRIES: int = 3
RETRY_DELAY: int = 1  # seconds between retries
TEMPERATURE: float = 2.0  # Higher temperature for more variable responses

# API Key
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables") 