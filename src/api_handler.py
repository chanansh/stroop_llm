import time
import random
from loguru import logger
from config import (
    MODEL, MAX_RETRIES, RETRY_DELAY
)
from mock_api_handler import MockAPIHandler
from base_api_handler import BaseAPIHandler

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

class RealAPIHandler(BaseAPIHandler):
    def __init__(self, client):
        super().__init__()
        self.client = client
    
    def _send_request(self, messages, max_tokens):
        """Send request to OpenAI API."""
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip().lower()
    
    def get_response(self, messages, max_tokens=10):
        """Get response from API with retry logic.
        
        Args:
            messages: List of message dictionaries for the API
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            tuple: (response_text, timing_info) where timing_info is a dict with:
                - total_time: Total time including retries
                - api_time: Time for successful API call only
                - retry_count: Number of retries needed
        """
        logger.debug(f"Getting response with max_tokens={max_tokens}")
        total_start = time.time()
        retry_count = 0
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{MAX_RETRIES}")
                api_start = time.time()
                result = self._send_request(messages, max_tokens)
                api_time = time.time() - api_start
                logger.debug(f"API response received: {result}")
                
                timing_info = {
                    "total_time": time.time() - total_start,
                    "api_time": api_time,
                    "retry_count": retry_count
                }
                return result, timing_info
                
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
                retry_count += 1
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
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
                model=MODEL,
                messages=messages,
                max_tokens=20
            )
            result = response.choices[0].message.content.strip()
            logger.debug(f"Feedback response received: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get feedback response: {str(e)}")
            return "Error processing feedback"
    
    def get_ping_time(self):
        """Measure network latency by timing a simple API call."""
        try:
            start_time = time.time()
            self._send_request([{"role": "user", "content": "ping"}], max_tokens=1)
            return time.time() - start_time
        except Exception as e:
            logger.warning(f"Failed to measure ping time: {str(e)}")
            return 0.0  # Return 0 if ping fails 