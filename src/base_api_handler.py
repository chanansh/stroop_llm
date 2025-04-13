from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from config import MAX_RETRIES, RETRY_DELAY
import time

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class BaseAPIHandler(ABC):
    """Base class for API handlers that provides common functionality for interacting with LLMs.
    
    This class implements the common patterns for API interaction while leaving the specific
    implementation of the API call to the concrete classes.
    """
    
    def __init__(self) -> None:
        """Initialize the API handler."""
        logger.debug(f"Initializing {self.__class__.__name__}")
    
    def validate_response(self, response: str) -> bool:
        """Validate that the response is in the expected format.
        
        Args:
            response: The response string to validate
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        is_valid = response in ["b", "m"]
        if not is_valid:
            logger.warning(f"Invalid response format: {response}")
        return is_valid
    
    @abstractmethod
    def _send_request(self, messages: List[Dict[str, Any]], max_tokens: int) -> str:
        """Send a request to the API and return the response.
        
        This method must be implemented by concrete classes to handle the specific
        API interaction.
        
        Args:
            messages: List of message dictionaries for the API
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            str: The API response
            
        Raises:
            Exception: If the API call fails
        """
        pass
    
    def get_response(self, messages: List[Dict[str, Any]]) -> Tuple[Dict[str, str], float, float, int]:
        """
        Get a response from the API with retry logic.
        
        Args:
            messages: List of message dictionaries containing the conversation history
            
        Returns:
            Tuple containing:
            - response: The API response message in the correct format
            - reaction_time: Time taken for the API call (excluding retries)
            - total_time: Total time including retries
            - retry_count: Number of retries needed
            
        Raises:
            APIError: If all retry attempts fail
        """
        retry_count = 0
        start_time = time.time()
        
        while retry_count <= MAX_RETRIES:
            try:
                if retry_count > 0:  # Only log if it's a retry
                    logger.warning(f"API call attempt {retry_count + 1}/{MAX_RETRIES + 1}")
                response_text = self._send_request(messages, max_tokens=10)  # Default max_tokens for color responses
                end_time = time.time()
                reaction_time = end_time - start_time
                total_time = reaction_time + (retry_count * RETRY_DELAY)
                
                # Return the response in the correct message format
                response_message = {
                    "role": "assistant",
                    "content": response_text
                }
                return response_message, reaction_time, total_time, retry_count
            except APIError as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    logger.warning(f"API call failed: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"All API call attempts failed after {MAX_RETRIES} retries")
                    raise
    
    def get_feedback_response(self, messages: List[Dict[str, Any]]) -> str:
        """Get a feedback response from the API.
        
        Args:
            messages: List of message dictionaries for the API
            
        Returns:
            str: The feedback response, or an error message if the call fails
        """
        logger.debug("Getting feedback response")
        try:
            result = self._send_request(messages, max_tokens=20)
            logger.debug(f"Feedback response received: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get feedback response: {str(e)}")
            return "Error processing feedback" 