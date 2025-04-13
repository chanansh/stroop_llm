import os
from loguru import logger
from experiment import run_experiment

# Configure logger
logger.remove()
logger.add("test_experiment.log", rotation="1 MB")
logger.add(lambda msg: print(msg.record["message"]), level="INFO")

def main():
    try:
        logger.info("Starting dry test of Stroop experiment...")
        
        # Create necessary directories
        os.makedirs("stimuli_images", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Run experiment in mock mode
        logger.info("Running experiment in mock mode...")
        run_experiment(client=None, use_mock=True)
        
        # Check results
        if os.path.exists("results"):
            logger.info("Results directory created successfully")
        if os.path.exists("stimuli_images"):
            logger.info("Stimuli images directory created successfully")
            
        logger.info("Dry test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dry test: {str(e)}")
        raise
    finally:
        logger.info("Test cleanup completed")

if __name__ == "__main__":
    main() 