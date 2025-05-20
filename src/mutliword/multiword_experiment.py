# run an experiment with chatgpt 4 as a participant.
# the model needs to reply to a text prompt and an image.
# the text prompt is: name the word color ink, ignore the word meaning. read line by line from left to right.
# the image is a grid of words with different colors.
# 50% are congruent, 50% are incongruent.
# in the congruent trials, the color of the word is the same as the color of the word in the image.
# in the incongruent trials, the color of the word is different from the color of the word in the image.

from datetime import datetime
import os
import random
import time
import json
import numpy as np
import openai
import pandas as pd
from mutliword.multiword_stimulus import MultipleWords
from mutliword.experiment_config import ExperimentConfig
import dotenv
from loguru import logger
from dataclasses import dataclass, asdict, field
from typing import Dict, List
import sys
from pprint import pprint
dotenv.load_dotenv()

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)



def get_words_and_colors(config: ExperimentConfig):
    words = []
    colors = []
    
    # Separate color words and neutral words
    color_words = [word for word, color in config.color_mapping.items() if color is not None]
    neutral_words = [word for word, color in config.color_mapping.items() if color is None]
    
    for _ in range(config.number_of_words_per_trial):
        # Decide if this should be a neutral word
        is_neutral = random.random() < config.neutral_word_probability
        
        if is_neutral:
            word = random.choice(neutral_words)
            # For neutral words, assign a random color
            color = random.choice([c for c in config.color_mapping.values() if c is not None])
        else:
            word = random.choice(color_words)
            # 50% are congruent, 50% are incongruent for color words
            congruent = random.random() < 0.5
            if congruent:
                color = config.color_mapping[word]
            else:
                # must be a different color
                color = random.choice(
                    [
                        c
                        for c in config.color_mapping.values()
                        if c is not None and c != config.color_mapping[word]
                    ]
                )
        words.append(word)
        colors.append(color)
    
    return words, colors

def strip_markdown_json(content: str) -> str:
    """Strip markdown code block from JSON response."""
    if content.startswith("```json") and content.endswith("```"):
        return content[7:-3].strip()
    return content

def save_trial_data(trial_data: dict, output_path: str, trial: int):
    """Save trial data to a JSON file."""
    trial_file = os.path.join(output_path, f"trial_{trial}.json")
    with open(trial_file, "w") as f:
        json.dump(trial_data, f, default=str)  # default=str to handle datetime objects

def run_trial(trial: int, config: ExperimentConfig):
    words, colors = get_words_and_colors(config)
    logger.info(f"Trial {trial} started, words: {words}, colors: {colors}")
    total_congruent = sum(1 for word, color in zip(words, colors) 
                         if config.color_mapping[word] is not None and color == config.color_mapping[word])
    total_incongruent = sum(1 for word, color in zip(words, colors) 
                           if config.color_mapping[word] is not None and color != config.color_mapping[word])
    total_neutral = sum(1 for word in words if config.color_mapping[word] is None)
    logger.info(f"total_congruent: {total_congruent}, total_incongruent: {total_incongruent}, total_neutral: {total_neutral}")
    logger.info(f"Generating stimulus...")
    stimulus = MultipleWords(
        words=words,
        colors=colors,
        words_per_row=config.words_per_row,
        char_spacing=config.char_spacing,
        line_spacing=config.line_spacing,
        font_size=config.font_size,
        font_name=config.font_name,
        background_color=config.background_color,
        alignment=config.alignment,
    )
    image_base64 = stimulus.get_base64()
    valid_response = False
    retries = 0
    response_json = None  # Initialize response_json to None
    logger.info(f"Sending request to OpenAI...")
    
    # API call with retry logic
    api_retries = 0
    retry_delay = config.initial_retry_delay
    
    while not valid_response and retries < config.max_retries:
        retries += 1
        while api_retries < config.max_api_retries:
            try:
                response = openai.chat.completions.create(
                    model=config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": config.prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                                },
                            ],
                        }
                    ],
                )
                # try to read the output json
                try:
                    content = response.choices[0].message.content
                    total_tokens = response.usage.total_tokens
                    content = strip_markdown_json(content)
                    response_json = json.loads(content)
                    valid_response = True
                    break  # Break the API retry loop on success
                except json.JSONDecodeError:
                    logger.warning(f"Invalid response: {response.choices[0].message.content}, trial {trial}, out of {config.max_retries} retries")
                    valid_response = False
                    break  # Break the API retry loop on invalid JSON
            except openai.RateLimitError as e:
                api_retries += 1
                if api_retries < config.max_api_retries:
                    logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Max API retries reached for trial {trial}, total tokens: {total_tokens}")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error in trial {trial}: {str(e)}")
                raise e
    logger.info(f"response: {response_json}")
    logger.info(f"saving stimulus...")
    stimulus.save(os.path.join(config.output_path, f"trial_{trial}.png"))
    len_response = len(response_json) if response_json is not None else 0
    len_stimulus = len(words)
    if len_response != len_stimulus:
        logger.warning(f"AI response length {len_response} does not match stimulus length {len_stimulus}")
    
    trial_data = {
        "trial_id": trial,
        "word": words,
        "color": colors,
        "response": response_json if response_json is not None else [],
        "trial_time": datetime.now(),
        "total_tokens": total_tokens,
        "prompt": config.prompt,
    }
    logger.info(trial_data)
    
    # Save trial data immediately
    save_trial_data(trial_data, os.path.join(config.output_path), trial)
    
    return trial_data

def run_experiment(config: ExperimentConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    logger.info(pprint(config))
    # save experiment parameters
    os.makedirs(config.output_path, exist_ok=True)
    with open(
        os.path.join(config.output_path, "experiment_parameters.json"), "w"
    ) as f:
        json.dump(asdict(config), f)
    
    experiment_data = []
    for trial in range(config.number_of_trials):
        logger.info(f"Running trial {trial} of {config.number_of_trials}")
        try:
            trial_data = run_trial(trial, config)
            experiment_data.append(trial_data)
        except Exception as e:
            logger.error(f"Error in trial {trial}: {str(e)}")
            # Save the experiment data we have so far
            with open(os.path.join(config.output_path, "experiment_data.json"), "w") as f:
                json.dump(experiment_data, f, default=str)
            raise e
    
    # Save the final experiment data
    with open(os.path.join(config.output_path, "experiment_data.json"), "w") as f:
        json.dump(experiment_data, f, default=str)
    
    return experiment_data

if __name__ == "__main__":
    config = ExperimentConfig()
    experiment_data = run_experiment(config)
