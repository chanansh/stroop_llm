import os
import random
import pandas as pd
from datetime import datetime
from itertools import product
from typing import List, Tuple
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
import json

from config import (
    PRACTICE_TRIALS, TEST_TRIALS, MESSAGES_PER_PRACTICE_TRIAL,
    WORDS, COLORS, KEY_MAPPING,
    IMAGE_DIR, LOG_FILE, RESULTS_DIR,
    DRY_RUN, NUM_PARTICIPANTS, SYSTEM_MESSAGE,
    MODEL, MAX_RETRIES, RETRY_DELAY,
    OPENAI_API_KEY, VIEWING_DISTANCE_CM,
    IMG_WIDTH, IMG_HEIGHT, FONT_SIZE,
    BACKGROUND_COLOR, FONT_NAME
)
from gpt4_old.trial_manager import TrialManager
from gpt4_old.image_handler import ImageHandler

# Configure logger to show debug messages
logger.remove()
logger.add(
    LOG_FILE,
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO"
)
logger.add(
    lambda msg: print(msg.record["message"]),
    level="INFO",
    format="{message}"
)

# Load environment variables
load_dotenv()

# OpenAI API setup
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def generate_trial_sequence(num_trials: int) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Generate a random sequence of trials.
    
    Args:
        num_trials: Number of trials to generate
        
    Returns:
        List of (word, color) tuples for the trials
    """
    if num_trials < 0:
        raise ValueError("Number of trials cannot be negative")
        
    # Generate all possible combinations
    all_stimuli = list(product(WORDS, COLORS))
    return [random.choice(all_stimuli) for _ in range(num_trials)]

def run_participant_experiment(participant_id, client, use_dryrun=False, output_file=None):
    """Run the experiment for a single participant."""
    logger.debug(f"Starting experiment for participant {participant_id}...")
    
    # Initialize components
    trial_manager = TrialManager(client=client, use_dryrun=use_dryrun)
    
    # Initialize messages with system prompt for this participant
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    logger.debug("System message initialized")
    logger.debug(f"Initial messages: {json.dumps(messages, indent=2)}")
    
    # Generate trials
    practice_trials = generate_trial_sequence(PRACTICE_TRIALS)
    main_trials = generate_trial_sequence(TEST_TRIALS)
    logger.debug(f"Generated {len(practice_trials)} practice trials and {len(main_trials)} main trials")
    
    # Initialize results list
    all_results = []
    
    # Print trial header
    trial_manager._log_trial_header()
    
    # Run practice trials
    logger.debug(f"Starting practice trials for participant {participant_id}")
    for trial, (word, color) in enumerate(practice_trials):
        logger.debug(f"\nPractice trial {trial + 1}: word={word}, color={color}")
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=True
        )
        all_results.append(trial_result)
        
        # Append trial result to CSV
        is_first_write = not os.path.exists(output_file)
        # Ensure all fields are included in the DataFrame
        trial_df = pd.DataFrame([trial_result])
        # Reorder columns to put alternative_sum in a logical position
        columns = [
            'participant_id', 'trial', 'word', 'color', 'response', 
            'correct_response', 'accuracy', 'retry_count',
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'response_logprob', 'second_best_response', 'second_best_logprob',
            'alternative_sum', 'condition'
        ]
        trial_df = trial_df[columns]
        trial_df.to_csv(
            output_file, 
            mode='a', 
            header=is_first_write, 
            index=False
        )
        logger.debug(f"Appended trial {trial + 1} (practice) to {os.path.abspath(output_file)}")
        
        logger.debug(f"Messages after trial: {json.dumps(messages, indent=2)}")
    
    # Run main trials
    logger.debug(f"Starting main trials for participant {participant_id}")
    for trial, (word, color) in enumerate(main_trials):
        logger.debug(f"\nMain trial {trial + 1}: word={word}, color={color}")
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=False
        )
        all_results.append(trial_result)
        
        # Append trial result to CSV
        is_first_write = not os.path.exists(output_file)
        # Ensure all fields are included in the DataFrame
        trial_df = pd.DataFrame([trial_result])
        # Reorder columns to put alternative_sum in a logical position
        columns = [
            'participant_id', 'trial', 'word', 'color', 'response', 
            'correct_response', 'accuracy', 'retry_count',
            'prompt_tokens', 'completion_tokens', 'total_tokens',
            'response_logprob', 'second_best_response', 'second_best_logprob',
            'alternative_sum', 'condition'
        ]
        trial_df = trial_df[columns]
        trial_df.to_csv(
            output_file, 
            mode='a', 
            header=is_first_write, 
            index=False
        )
        logger.debug(f"Appended trial {trial + 1} (main) to {os.path.abspath(output_file)}")
        
        logger.debug(f"Messages after trial: {json.dumps(messages, indent=2)}")
    
    return all_results

def run_experiment(client=None, use_dryrun=False):
    """Run the complete experiment for all participants."""
    logger.debug("Starting experiment...")
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate the output filename at the start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"experiment_{timestamp}.csv")
    config_file = os.path.join(RESULTS_DIR, f"experiment_{timestamp}.json")
    logger.info(f"All results will be saved to: {os.path.abspath(output_file)}")
    
    # Save experiment configuration
    config = {
        "practice_trials": PRACTICE_TRIALS,
        "test_trials": TEST_TRIALS,
        "messages_per_practice_trial": MESSAGES_PER_PRACTICE_TRIAL,
        "words": WORDS,
        "colors": COLORS,
        "key_mapping": {str(k): v for k, v in KEY_MAPPING.items()},
        "dry_run": DRY_RUN,
        "num_participants": NUM_PARTICIPANTS,
        "system_message": SYSTEM_MESSAGE,
        "model": MODEL,
        "max_retries": MAX_RETRIES,
        "retry_delay": RETRY_DELAY,
        "viewing_distance_cm": VIEWING_DISTANCE_CM,
        "img_width": IMG_WIDTH,
        "img_height": IMG_HEIGHT,
        "font_size": FONT_SIZE,
        "background_color": BACKGROUND_COLOR,
        "font_name": FONT_NAME
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Experiment configuration saved to: {os.path.abspath(config_file)}")
    
    for participant_id in range(1, NUM_PARTICIPANTS + 1):
        participant_results = run_participant_experiment(
            participant_id, 
            client, 
            use_dryrun,
            output_file=output_file
        )
        logger.info(f"Completed participant {participant_id}/{NUM_PARTICIPANTS}")
    
    logger.info("Experiment completed and all results saved")
    
    return pd.read_csv(output_file).to_dict('records')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="Run in dry run mode", default=False)
    args = parser.parse_args()
    
    # Override DRY_RUN if --dryrun flag is provided
    if args.dryrun:
        logger.debug("Running in dry run mode")
        client = None
    else:
        logger.debug("Running with real OpenAI API")
        client = OpenAI()
    
    run_experiment(client=client, use_dryrun=args.dryrun)
