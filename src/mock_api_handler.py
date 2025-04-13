import random
import time
from loguru import logger
from .config import ExperimentConfig

class MockAPIHandler:
    def __init__(self):
        logger.info("Initializing Mock API Handler")
        self.config = ExperimentConfig
        self.response_times = []  # Track response times for realism
        self.error_rate = 0.1  # 10% error rate
        logger.debug(f"Set error rate to {self.error_rate}")
    
    def get_response(self, messages, max_tokens=10):
        """Simulate API response with realistic timing."""
        logger.debug("Processing mock API request")
        
        # Generate random response
        response = random.choice(["b", "m"])
        
        # Return mock response
        return response
    
    def validate_response(self, response):
        """Validate that response is valid."""
        is_valid = response in ["b", "m"]
        if not is_valid:
            logger.warning(f"Invalid mock response: {response}")
        return is_valid
    
    def get_feedback_response(self, messages):
        """Simulate feedback response."""
        logger.debug("Generating mock feedback response")
        latency = random.uniform(0.1, 0.3)
        logger.debug(f"Simulating feedback latency: {latency:.3f}s")
        time.sleep(latency)
        
        response = random.choice([
            "Thank you for the feedback!",
            "I understand now.",
            "I'll try to be more accurate.",
            "Got it, thanks!",
            "I'll focus on the color next time."
        ])
        logger.debug(f"Generated mock feedback: {response}")
        return response 