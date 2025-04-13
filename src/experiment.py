import os
import time
import random
import pandas as pd
from itertools import product
from typing import List, Tuple
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from config import (
    PRACTICE_TRIALS, STIMULI_COUNT, MESSAGES_PER_PRACTICE_TRIAL,
    WORDS, COLORS, KEY_MAPPING,
    IMAGE_DIR, LOG_FILE, RESULTS_DIR,
    DRY_RUN, NUM_PARTICIPANTS, SYSTEM_MESSAGE,
    MODEL, MAX_RETRIES, RETRY_DELAY,
    OPENAI_API_KEY
)
from trial_manager import TrialManager
from utils import measure_ping
from stimulus_generator import StimulusGenerator

# Configure logger
logger.remove()
logger.add(
    LOG_FILE,
    rotation="1 MB",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
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

def run_trial(*, client, trial, word, color, participant_id, messages, practice=False):
    """Run a single trial, measuring response and providing feedback if practice."""
    # Initialize trial manager
    trial_manager = TrialManager(client=client, use_mock=DRY_RUN)
    
    # Run the trial and get results
    trial_result = trial_manager.run_trial(
        trial=trial,
        word=word,
        color=color,
        participant_id=participant_id,
        messages=messages,
        practice=practice
    )
    
    return trial_result

def run_participant_experiment(participant_id, client, use_mock=False):
    """Run the experiment for a single participant."""
    logger.debug(f"Starting experiment for participant {participant_id}...")
    
    # Initialize components
    trial_manager = TrialManager(client=client, use_mock=use_mock)
    
    # Initialize messages with system prompt for this participant
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    logger.debug("System message initialized")
    
    # Generate trials
    practice_trials = generate_trial_sequence(PRACTICE_TRIALS)
    main_trials = generate_trial_sequence(STIMULI_COUNT)
    logger.debug(f"Generated {len(practice_trials)} practice trials and {len(main_trials)} main trials")
    
    # Run practice trials
    logger.debug(f"Starting practice trials for participant {participant_id}")
    practice_results = []
    for trial, (word, color) in enumerate(practice_trials):
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=True
        )
        practice_results.append(trial_result)
    
    # Run main trials
    logger.debug(f"Starting main trials for participant {participant_id}")
    main_results = []
    for trial, (word, color) in enumerate(main_trials):
        trial_result = trial_manager.run_trial(
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            messages=messages,
            practice=False
        )
        main_results.append(trial_result)
    
    # Save results
    all_results = practice_results + main_results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULTS_DIR}/participant_{participant_id}.csv", index=False)
    logger.debug(f"Results saved for participant {participant_id}")
    
    return all_results

def run_experiment(client=None, use_mock=False):
    """Run the complete experiment for all participants."""
    logger.debug("Starting experiment...")
    all_results = []
    
    for participant_id in range(1, NUM_PARTICIPANTS + 1):
        participant_results = run_participant_experiment(participant_id, client, use_mock)
        all_results.extend(participant_results)
    
    # Save combined results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(f"{RESULTS_DIR}/all_participants.csv", index=False)
    logger.debug("Experiment completed and all results saved")
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true", help="Run in dry run mode", default=True)
    args = parser.parse_args()
    
    # Override DRY_RUN if --dryrun flag is provided
    if args.dryrun:
        logger.debug("Running in dry run mode")
        client = None
        dryrun = True
    else:
        logger.debug("Running with real OpenAI API")
        client = OpenAI()
        dryrun = False
    
    run_experiment(client=client, use_mock=args.dryrun)
