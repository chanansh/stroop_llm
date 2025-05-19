import argparse
import os
from loguru import logger
from gpt4_old.experiment import run_experiment
from config import ExperimentConfig

def setup_logging(log_file: str = "experiment.log") -> None:
    """Configure logging for the experiment.
    
    Args:
        log_file: Path to the log file
    """
    logger.remove()
    logger.add(
        log_file,
        rotation="1 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )
    logger.add(
        lambda msg: print(msg.record["message"]),
        level="INFO",
        format="{level: <8} | {message}"
    )

def create_directories() -> None:
    """Create necessary directories for the experiment."""
    os.makedirs(ExperimentConfig.IMAGE_DIR, exist_ok=True)
    os.makedirs(ExperimentConfig.RESULTS_DIR, exist_ok=True)
    logger.info(f"Created directories: {ExperimentConfig.IMAGE_DIR}, {ExperimentConfig.RESULTS_DIR}")

def main() -> None:
    """Main entry point for the experiment CLI."""
    parser = argparse.ArgumentParser(description="Run a Stroop effect experiment with LLMs")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no API calls)"
    )
    parser.add_argument(
        "--participants",
        type=int,
        default=ExperimentConfig.NUM_PARTICIPANTS,
        help="Number of participants to run"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="experiment.log",
        help="Path to the log file"
    )
    
    args = parser.parse_args()
    
    # Update config if needed
    if args.participants != ExperimentConfig.NUM_PARTICIPANTS:
        ExperimentConfig.NUM_PARTICIPANTS = args.participants
        logger.info(f"Updated number of participants to {args.participants}")
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Create directories
    create_directories()
    
    try:
        logger.info("Starting Stroop experiment...")
        logger.info(f"Mode: {'Mock' if args.mock else 'Real'}")
        logger.info(f"Participants: {ExperimentConfig.NUM_PARTICIPANTS}")
        
        # Run experiment
        results = run_experiment(client=None, use_mock=args.mock)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved in {ExperimentConfig.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 