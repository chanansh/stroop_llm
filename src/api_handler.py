import time
from loguru import logger
from .config import ExperimentConfig
from .mock_api_handler import MockAPIHandler

class APIHandlerFactory:
    @staticmethod
    def create_handler(client=None, use_mock=False):
        logger.info(f"Creating {'mock' if use_mock else 'real'} API handler")
        if use_mock:
            return MockAPIHandler()
        if client is None:
            logger.error("No client provided for real API handler")
            raise ValueError("Client must be provided for real API handler")
        return RealAPIHandler(client)

class RealAPIHandler:
    def __init__(self, client):
        logger.info("Initializing Real API Handler")
        self.client = client
        self.config = ExperimentConfig
    
    def get_response(self, messages, max_tokens=10):
        """Get response from API with retry logic."""
        logger.debug(f"Getting response with max_tokens={max_tokens}")
        for attempt in range(self.config.MAX_RETRIES):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.config.MAX_RETRIES}")
                response = self.client.chat.completions.create(
                    model=self.config.MODEL,
                    messages=messages,
                    max_tokens=max_tokens
                )
                result = response.choices[0].message.content.strip().lower()
                logger.debug(f"API response received: {result}")
                return result
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.info(f"Retrying in {self.config.RETRY_DELAY} seconds...")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.critical("All retry attempts failed")
                    raise
    
    def validate_response(self, response):
        """Validate that response is valid."""
        is_valid = response in ["b", "m"]
        if not is_valid:
            logger.warning(f"Invalid response format: {response}")
        return is_valid
    
    def get_feedback_response(self, messages):
        """Get feedback response from API."""
        logger.debug("Getting feedback response")
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL,
                messages=messages,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"Feedback response received: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get feedback response: {str(e)}")
            return "Error processing feedback" 