import os
import time
import random
import pandas as pd
from itertools import product
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
import base64
from PIL import Image, ImageDraw, ImageFont
import math
from dataclasses import dataclass

from src.config import ExperimentConfig
from src.image_handler import ImageHandler
from src.trial_manager import TrialManager

# Configure logger
logger.remove()
logger.add(
    "experiment.log",
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
)
logger.add(
    lambda msg: print(msg.record["message"]),
    level="INFO",
    format="{level: <8} | {message}"
)

# Load environment variables
load_dotenv()

# OpenAI API setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

@dataclass
class ExperimentConfig:
    WORDS = ["BLUE", "RED", "XXXX"]
    COLORS = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    BACKGROUND_COLOR = (192, 192, 192)  # Silver
    FONT_NAME = "Courier New"
    PRACTICE_TRIALS = 10
    STIMULI_COUNT = 6*60  # 6 blocks * 60 trials
    IMG_WIDTH = 800
    IMG_HEIGHT = 400
    VIEWING_DISTANCE_CM = 63.5  # Viewing distance in cm
    DPI = 96  # Approximate screen DPI (dots per inch)
    CM_TO_PIXELS = DPI / 2.54  # Conversion factor
    HEIGHT_ANGLE = (2.28 + 3.16) / 2  # Average height visual angle
    WIDTH_ANGLE = (10.44 + 10.8) / 2  # Average width visual angle
    FONT_SIZE = int(2 * VIEWING_DISTANCE_CM * math.tan(math.radians(HEIGHT_ANGLE / 2)) * CM_TO_PIXELS)
    IMAGE_DIR = "stimuli_images"
    LOG_FILE = "experiment.log"
    DRY_RUN = False  # Set to True for testing without OpenAI API
    NUM_PARTICIPANTS = 3
    RESULTS_DIR = "results"
    SYSTEM_MESSAGE = """You are participating in a Stroop experiment. Your task is to identify the color of the text, not the word itself.
    
Rules:
1. Respond with 'b' if the text color is blue
2. Respond with 'm' if the text color is red
3. Ignore the meaning of the word, focus only on its color
4. Respond as quickly and accurately as possible

During practice trials, you will receive feedback on your responses. Use this feedback to improve your performance."""
    MEMORY_LIMIT = 10  # Number of messages to keep in memory
    MODEL = "gpt-4-vision-preview"  # Updated to use vision model
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    IMAGE_QUALITY = "high"  # high, low, or auto
    IMAGE_DETAIL = "high"  # high or low

# Key mappings
KEY_MAPPING = {
    (0, 0, 255): "b",  # Blue color -> 'b'
    (255, 0, 0): "m"  # Red color -> 'm'
}

# Generate all stimuli conditions
STIMULI = list(product(ExperimentConfig.WORDS, ExperimentConfig.COLORS))

# Create directory for images
os.makedirs(ExperimentConfig.IMAGE_DIR, exist_ok=True)

import socket
import time

def measure_ping(host="api.openai.com", port=443, count=5):
    """
    Measures the approximate network latency (ping) by timing a TCP handshake.
    This method avoids using system-dependent 'ping' commands.
    """
    latencies = []

    for _ in range(count):
        try:
            # Create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # Set timeout for connection attempt
            
            start_time = time.time()  # Start timing
            sock.connect((host, port))  # Open TCP connection
            end_time = time.time()  # End timing
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

            sock.close()  # Close connection
        except socket.error:
            latencies.append(None)  # Store None if the connection fails

    # Filter out failed attempts
    latencies = [lat for lat in latencies if lat is not None]

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        return avg_latency
    else:
        return None
    
def get_image_filename(word, color):
    return f"{ExperimentConfig.IMAGE_DIR}/{word}_{color[0]}_{color[1]}_{color[2]}.png"

def create_stimulus_images():
    """Generate stimulus images if they don't already exist."""
    logger.info("Generating stimulus images...")
    for word, color in STIMULI:
        filename = get_image_filename(word, color)
        if not os.path.exists(filename):
            img = Image.new("RGB", (ExperimentConfig.IMG_WIDTH, ExperimentConfig.IMG_HEIGHT), ExperimentConfig.BACKGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(ExperimentConfig.FONT_NAME, ExperimentConfig.FONT_SIZE)  # Use Courier New
            except:
                font = ImageFont.load_default()
            text_size = draw.textbbox((0, 0), word, font=font)
            text_x = (ExperimentConfig.IMG_WIDTH - text_size[2]) // 2
            text_y = (ExperimentConfig.IMG_HEIGHT - text_size[3]) // 2
            draw.text((text_x, text_y), word, fill=color, font=font)
            img.save(filename)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_trial(*, client, trial, word, color, participant_id, messages, practice=False):
    """Run a single trial, measuring response and providing feedback if practice."""
    logger.info(f"Participant {participant_id} - Trial {trial+1}: Stimulus - {word} ({color}), Practice - {practice}")
    img_filename = get_image_filename(word, color)
    start_time = time.time()
    
    if ExperimentConfig.DRY_RUN:
        gpt_response = random.choice(["b", "m"])
    else:
        base64_image = encode_image(img_filename)
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": ExperimentConfig.IMAGE_DETAIL
                }
            }
        ]
        
        # Add the image to messages
        messages.append({"role": "user", "content": content})
        
        # Implement retry logic
        for attempt in range(ExperimentConfig.MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=ExperimentConfig.MODEL,
                    messages=messages,
                    max_tokens=10  # Limit response length
                )
                gpt_response = response.choices[0].message.content.strip().lower()
                
                # Validate response
                if gpt_response in ["b", "m"]:
                    break
                else:
                    logger.warning(f"Invalid response '{gpt_response}' on attempt {attempt + 1}")
                    messages.append({"role": "user", "content": "Invalid response. Please respond with 'b' or 'm'."})
                    if attempt < ExperimentConfig.MAX_RETRIES - 1:
                        time.sleep(ExperimentConfig.RETRY_DELAY)
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < ExperimentConfig.MAX_RETRIES - 1:
                    time.sleep(ExperimentConfig.RETRY_DELAY)
                else:
                    raise
    
    # Measure response time and accuracy
    end_time = time.time()
    rt = end_time - start_time
    ping_time = measure_ping(count=1)
    
    # Determine accuracy
    correct_response = KEY_MAPPING[color]
    accuracy = 1 if gpt_response == correct_response else 0
    
    # Provide feedback if practice
    if practice:
        feedback = "Correct!" if accuracy else f"Incorrect. The correct response is '{correct_response}'"
        messages.append({"role": "user", "content": feedback})
        logger.info(f"Feedback: {feedback}")
        
        if not ExperimentConfig.DRY_RUN:
            try:
                response = client.chat.completions.create(
                    model=ExperimentConfig.MODEL,
                    messages=messages,
                    max_tokens=20
                )
                feedback_response = response.choices[0].message.content.strip()
                logger.info(f"Feedback response: {feedback_response}")
            except Exception as e:
                logger.error(f"Failed to get feedback response: {str(e)}")
                feedback_response = "Error processing feedback"
        else:
            feedback_response = "Thank you for your feedback!"
    else:
        feedback = None
        feedback_response = None
    
    # Log trial results
    trial_result = {
        "Participant": participant_id,
        "Trial": trial + 1,
        "Word": word,
        "Color": color,
        "GPT Response": gpt_response,
        "Correct Response": correct_response,
        "Reaction Time (s)": rt,
        "Accuracy": accuracy,
        "Feedback": feedback,
        "Feedback Response": feedback_response,
        "Ping Time (ms)": ping_time
    }
    logger.info(trial_result)
    return trial_result

def run_participant_experiment(participant_id, client=None, use_mock=False):
    """Run experiment for a single participant."""
    logger.info(f"Starting experiment for participant {participant_id}...")
    results = []
    
    # Initialize components
    trial_manager = TrialManager(client=client, use_mock=use_mock)
    
    # Initialize messages with system prompt
    messages = [{"role": "system", "content": ExperimentConfig.SYSTEM_MESSAGE}]
    logger.debug("System message initialized")
    
    # Generate trials
    practice_trials = [random.choice(list(product(ExperimentConfig.WORDS, ExperimentConfig.COLORS))) 
                      for _ in range(ExperimentConfig.PRACTICE_TRIALS)]
    main_trials = [random.choice(list(product(ExperimentConfig.WORDS, ExperimentConfig.COLORS))) 
                  for _ in range(ExperimentConfig.STIMULI_COUNT)]
    logger.debug(f"Generated {len(practice_trials)} practice trials and {len(main_trials)} main trials")
    
    # Run practice trials
    logger.info(f"Starting practice trials for participant {participant_id}")
    for trial, (word, color) in enumerate(practice_trials):
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=True
        )
        results.append(trial_result)
        
        # Maintain context length
        if len(messages) > ExperimentConfig.PRACTICE_TRIALS + 1:
            messages = [messages[0]] + messages[-(ExperimentConfig.PRACTICE_TRIALS):]
            logger.debug(f"Trimmed message context to {len(messages)} messages")
    
    # Run main trials
    logger.info(f"Starting main trials for participant {participant_id}")
    for trial, (word, color) in enumerate(main_trials, start=len(practice_trials)):
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=False
        )
        results.append(trial_result)
        
        # Maintain context length
        if len(messages) > ExperimentConfig.MEMORY_LIMIT:
            messages = [messages[0]] + messages[-(ExperimentConfig.MEMORY_LIMIT-1):]
            logger.debug(f"Trimmed message context to {len(messages)} messages")
    
    # Save results
    df = pd.DataFrame(results)
    os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
    result_filename = f"{ExperimentConfig.RESULTS_DIR}/stroop_experiment_results_{participant_id}.csv"
    df.to_csv(result_filename, index=False)
    logger.info(f"Results saved to {result_filename}")
    logger.info(f"Experiment completed for participant {participant_id}")

def run_experiment(client=None, use_mock=False):
    """Run the experiment for multiple participants."""
    logger.info("Starting Stroop experiment")
    image_handler = ImageHandler()
    image_handler.create_stimulus_images()
    
    for participant_id in range(1, ExperimentConfig.NUM_PARTICIPANTS + 1):
        run_participant_experiment(
            participant_id=participant_id,
            client=client,
            use_mock=use_mock
        )
    logger.info("Experiment completed for all participants")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Stroop experiment')
    parser.add_argument('--dryrun', action='store_true', help='Run in dry run mode without using OpenAI API')
    args = parser.parse_args()
    
    if args.dryrun:
        logger.info("Running in dry run mode")
        client = None
    else:
        logger.info("Running with real OpenAI API")
        client = OpenAI(api_key=ExperimentConfig.OPENAI_API_KEY)
    
    run_experiment(client=client, use_mock=args.dryrun)
