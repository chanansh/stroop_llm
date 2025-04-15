import time
import random
from loguru import logger
from base_api_handler import BaseAPIHandler
from config import RESPONSE_BUTTON_BLUE, RESPONSE_BUTTON_RED

class MockAPIHandler(BaseAPIHandler):
    def __init__(self):
        super().__init__()
        self.responses = [RESPONSE_BUTTON_BLUE, RESPONSE_BUTTON_RED]
        self.current_response = 0
    
    def _send_request(self, messages, max_tokens=1):
        """Mock API request that alternates between blue and red responses."""
        # Get the next response
        response = self.responses[self.current_response]
        self.current_response = (self.current_response + 1) % len(self.responses)
        
        # Mock usage data
        usage = type('Usage', (), {
            'prompt_tokens': 10,
            'completion_tokens': 1,
            'total_tokens': 11
        })
        
        # Mock logprobs data
        logprobs_data = {
            'content': {
                'token': response,
                'logprob': -0.5,
                'top_logprobs': [
                    {'token': response, 'logprob': -0.5},
                    {'token': RESPONSE_BUTTON_RED if response == RESPONSE_BUTTON_BLUE else RESPONSE_BUTTON_BLUE, 'logprob': -1.0}
                ]
            }
        }
        
        return response, usage, logprobs_data
    
    def get_response(self, messages, max_tokens=1):
        """Get mock response."""
        response, usage, logprobs = self._send_request(messages, max_tokens)
        
        # Create response message
        response_message = {
            "role": "assistant",
            "content": response
        }
        
        return (
            response_message,
            0,  # retry_count
            usage,  # token_usage
            logprobs  # log probabilities
        )
    
    def validate_response(self, response):
        """Validate that response is valid."""
        return response in [RESPONSE_BUTTON_BLUE, RESPONSE_BUTTON_RED]
    
    def get_feedback_response(self, messages):
        """Get mock feedback response."""
        return "Feedback received" 