import random
import time
import openai
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import math
from loguru import logger
from dataclasses import dataclass
from itertools import product

# OpenAI API setup (Replace with your API key)
OPENAI_API_KEY = "your-api-key"
openai.api_key = OPENAI_API_KEY

@dataclass
class ExperimentConfig:
    WORDS = ["BLUE", "RED", "XXXX"]
    COLORS = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    BACKGROUND_COLOR = (192, 192, 192)  # Silver
    FONT_NAME = "Courier New"
    STIMULI_COUNT = 360  # 6 blocks * 60 trials
    PRACTICE_TRIALS = 10
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
    DRY_RUN = True  # Set to True for testing without OpenAI API
    NUM_PARTICIPANTS = 3
    RESULTS_DIR = "results"

# Logging setup
logger.add(ExperimentConfig.LOG_FILE, rotation="1 MB")

# Key mappings
KEY_MAPPING = {
    (0, 0, 255): "b",  # Blue color -> 'b'
    (255, 0, 0): "m"  # Red color -> 'm'
}

# Generate all stimuli conditions
STIMULI = list(product(ExperimentConfig.WORDS, ExperimentConfig.COLORS))

# Create directory for images
os.makedirs(ExperimentConfig.IMAGE_DIR, exist_ok=True)

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

def run_trial(trial, word, color, participant_id, messages, practice=False):
    """Run a single trial, measuring response and providing feedback if practice."""
    logger.info(f"Participant {participant_id} - Trial {trial+1}: Stimulus - {word} ({color}), Practice - {practice}")
    img_filename = get_image_filename(word, color)
    start_time = time.time()
    
    messages.append({"role": "user", "content": "Observe the following image and respond with 'b' if the text color is blue, 'm' if the text color is red."})
    
    if ExperimentConfig.DRY_RUN:
        gpt_response = random.choice(["b", "m"])
    else:
        with open(img_filename, "rb") as img_file:
            messages.append({"role": "user", "image": img_file.read()})
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages
            )
            gpt_response = response["choices"][0]["message"]["content"].strip().lower()
    
    end_time = time.time()
    rt = end_time - start_time
    correct_response = KEY_MAPPING[color]
    accuracy = 1 if gpt_response == correct_response else 0
    
    feedback = "Correct!" if accuracy else f"Incorrect. The correct response is '{correct_response}'"
    messages.append({"role": "user", "content": feedback})
    
    if practice:
        logger.info(f"Feedback: {feedback}")
        if not ExperimentConfig.DRY_RUN:
            openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages
            )


    trial_result = {
        "Participant": participant_id,
        "Trial": trial + 1,
        "Word": word,
        "Color": color,
        "GPT Response": gpt_response,
        "Correct Response": correct_response,
        "Reaction Time (s)": rt,
        "Accuracy": accuracy
    }
    
    return trial_result

def run_participant_experiment(participant_id):
    """Run experiment for a single participant with persistent chat memory."""
    logger.info(f"Starting experiment for participant {participant_id}...")
    results = []
    trials = [random.choice(STIMULI) for _ in range(ExperimentConfig.PRACTICE_TRIALS + ExperimentConfig.STIMULI_COUNT)]
    messages = [
        {"role": "system", "content": "You are a paid participant in a psychological experiment. Your task is to carefully follow the given instructions and provide accurate responses."}
    ]
    
    for trial, (word, color) in enumerate(trials):
        practice = trial < ExperimentConfig.PRACTICE_TRIALS
        results.append(run_trial(trial, word, color, participant_id, messages, practice))
    
    df = pd.DataFrame(results)
    results_dir = ExperimentConfig.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    result_filename = f"{results_dir}/stroop_experiment_results_{participant_id}.csv"
    df.to_csv(result_filename, index=False)
    logger.info(f"Experiment completed for participant {participant_id}. Results saved.")

def run_experiment(num_participants):
    """Run the experiment for multiple participants."""
    create_stimulus_images()
    for participant_id in range(1, num_participants + 1):
        run_participant_experiment(participant_id)

if __name__ == "__main__":
    run_experiment(ExperimentConfig.NUM_PARTICIPANTS)
