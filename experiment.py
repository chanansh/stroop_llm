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
import base64

# TODO: add memory to the api so there will be learning. 
# OpenAI API setup (Replace with your API key)
OPENAI_API_KEY = "your-api-key"
openai.api_key = OPENAI_API_KEY

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
    DRY_RUN = True  # Set to True for testing without OpenAI API
    NUM_PARTICIPANTS = 3
    RESULTS_DIR = "results"
    SYSTEM_MESSAGE = "You are a paid participant in a psychological experiment. Your task is to carefully follow the given instructions and provide accurate responses."
    MEMORY_LIMIT = 10  # Number of messages to keep in memory
    INSTRUCTIONS = "Observe the following image and respond with 'b' if the text color is blue, 'm' if the text color is red."
    MODEL = "gpt-4o"
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

def run_trial(*,client, trial, word, color, participant_id, messages, practice=False):
    """Run a single trial, measuring response and providing feedback if practice."""
    logger.info(f"Participant {participant_id} - Trial {trial+1}: Stimulus - {word} ({color}), Practice - {practice}")
    img_filename = get_image_filename(word, color)
    start_time = time.time()
    if ExperimentConfig.DRY_RUN:
        gpt_response = random.choice(["b", "m"])
    else:
        base64_image = encode_image(img_filename)
        instructions = ExperimentConfig.INSTRUCTIONS
        content = [
                    { "type": "text", "text": instructions },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ]
        messages.append({"role": "user", "content": content})
        
        response = client.chat.completions.create(
            model=ExperimentConfig.MODEL,
            messages=messages
        )
        gpt_response = response["choices"][0]["message"]["content"].strip().lower()
        attempts_left = 3
        while (gpt_response not in ["b", "m"]) and attempts_left > 0:
            attempts_left -= 1
            messages.append({"role": "user", "content": "Invalid response. Please respond with 'b' or 'm'."})
            response = client.chat.completions.create(
                model=ExperimentConfig.MODEL,
                messages=messages
            )
            gpt_response = response["choices"][0]["message"]["content"].strip().lower()
    # Measure response time and accuracy
    end_time = time.time()
    rt = end_time - start_time
    ping_time = measure_ping(count=1) # Measure network latency to reduce variance in response time measurements
    # Determine accuracy
    correct_response = KEY_MAPPING[color]
    accuracy = 1 if gpt_response == correct_response else 0
    
    # Provide feedback if practice
    if practice:
        feedback = "Correct!" if accuracy else f"Incorrect. The correct response is '{correct_response}'"
        messages.append({"role": "user", "content": feedback})
        logger.info(f"Feedback: {feedback}")
        if not ExperimentConfig.DRY_RUN:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages
            )
            feedback_response = response["choices"][0]["message"]["content"].strip()
            logger.info(f"Feedback response: {feedback_response}")
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

def run_participant_experiment(participant_id, client):
    """Run experiment for a single participant with persistent chat memory."""
    logger.info(f"Starting experiment for participant {participant_id}...")
    results = []
    total_num_trials = ExperimentConfig.STIMULI_COUNT + ExperimentConfig.PRACTICE_TRIALS
    trials = [random.choice(STIMULI) for _ in range(total_num_trials)]
    messages = [
        {"role": "system", "content": ExperimentConfig.SYSTEM_MESSAGE}
    ] # init messages with system message
    
    for trial, (word, color) in enumerate(trials):
        practice = trial < ExperimentConfig.PRACTICE_TRIALS
        trial_result = run_trial(
            client = client,
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=practice
        )
        results.append(trial_result)
    num_messages = len(messages)
    if num_messages > ExperimentConfig.MEMORY_LIMIT:
        messages = [messages[0]] + messages[-ExperimentConfig.MEMORY_LIMIT:]  # Keep only the last messages
    df = pd.DataFrame(results)
    results_dir = ExperimentConfig.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    result_filename = f"{results_dir}/stroop_experiment_results_{participant_id}.csv"
    df.to_csv(result_filename, index=False)
    logger.info(f"Experiment completed for participant {participant_id}. Results saved.")

def run_experiment(client):
    """Run the experiment for multiple participants."""
    create_stimulus_images()
    for participant_id in range(1, ExperimentConfig.NUM_PARTICIPANTS + 1):
        run_participant_experiment(client=client, participant_id=participant_id)

if __name__ == "__main__":
    from openai import OpenAI
    if ExperimentConfig.DRY_RUN:
        logger.warning("Running in dry run mode. No API calls will be made.")
        client = None
    else:
        client  = OpenAI(organization=None, project=None)
    run_experiment(client)
