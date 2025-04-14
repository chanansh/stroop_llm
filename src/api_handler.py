import time
import json
from typing import Dict, List, Optional, Tuple
from loguru import logger
from openai import OpenAI
from config import (
    MODEL, MAX_RETRIES, RETRY_DELAY,
    SYSTEM_MESSAGE, TEMPERATURE
)
from mock_api_handler import MockAPIHandler
from base_api_handler import BaseAPIHandler

class APIHandlerFactory:
    @staticmethod
    def create_handler(client=None, use_dryrun=False):
        logger.info(f"Creating {'dry run' if use_dryrun else 'real'} API handler")
        if use_dryrun:
            return MockAPIHandler()
        if client is None:
            logger.error("No client provided for real API handler")
            raise ValueError("Client must be provided for real API handler")
        return RealAPIHandler(client)

class RealAPIHandler(BaseAPIHandler):
    def __init__(self, client):
        super().__init__()
        self.client = client
    
    def measure_rtt(self):
        """Measure round-trip time using a minimal API call."""
        try:
            start_time = time.time()
            # Use a minimal message that should get a quick response
            self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "."}],
                max_tokens=1
            )
            rtt = time.time() - start_time
            logger.debug(f"API RTT: {rtt:.3f}s")
            return rtt
        except Exception as e:
            logger.warning(f"Failed to measure API RTT: {str(e)}")
            return 0.0
    
    def _send_request(self, messages, max_tokens=1):
        """Send request to OpenAI API."""
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip().lower(), response.usage
    
    def get_response(self, messages, max_tokens=1):
        """Get response from API with retries."""
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{MAX_RETRIES}")
                start_time = time.time()
                response, usage = self._send_request(messages, max_tokens)
                end_time = time.time()
                
                # Create response message
                response_message = {
                    "role": "assistant",
                    "content": response
                }
                
                return (
                    response_message,
                    end_time - start_time,  # reaction_time
                    end_time - start_time,  # total_time
                    attempt,  # retry_count
                    usage  # token_usage
                )
            except Exception as e:
                logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
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
                temperature=TEMPERATURE,
                max_tokens=1
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