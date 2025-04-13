import os
import math
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ExperimentConfig:
    # Stimulus parameters
    WORDS = ["BLUE", "RED", "XXXX"]
    COLORS = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    BACKGROUND_COLOR = (192, 192, 192)  # Silver
    FONT_NAME = "Courier New"
    
    # Experiment parameters
    PRACTICE_TRIALS = 10
    STIMULI_COUNT = 6*60  # 6 blocks * 60 trials
    NUM_PARTICIPANTS = 3
    
    # Image parameters
    IMG_WIDTH = 800
    IMG_HEIGHT = 400
    VIEWING_DISTANCE_CM = 63.5  # Viewing distance in cm
    DPI = 96  # Approximate screen DPI (dots per inch)
    CM_TO_PIXELS = DPI / 2.54  # Conversion factor
    HEIGHT_ANGLE = (2.28 + 3.16) / 2  # Average height visual angle
    WIDTH_ANGLE = (10.44 + 10.8) / 2  # Average width visual angle
    FONT_SIZE = int(2 * VIEWING_DISTANCE_CM * math.tan(math.radians(HEIGHT_ANGLE / 2)) * CM_TO_PIXELS)
    
    # File paths
    IMAGE_DIR = "stimuli_images"
    LOG_FILE = "experiment.log"
    RESULTS_DIR = "results"
    
    # API settings
    DRY_RUN = False
    MODEL = "gpt-4-vision-preview"
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    IMAGE_QUALITY = "high"
    IMAGE_DETAIL = "high"
    MEMORY_LIMIT = 10
    
    # System message
    SYSTEM_MESSAGE = """You are participating in a Stroop experiment. Your task is to identify the color of the text, not the word itself.
    
Rules:
1. Respond with 'b' if the text color is blue
2. Respond with 'm' if the text color is red
3. Ignore the meaning of the word, focus only on its color
4. Respond as quickly and accurately as possible

During practice trials, you will receive feedback on your responses. Use this feedback to improve your performance."""
    
    # API Key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables") 