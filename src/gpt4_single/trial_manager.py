import os
import time
import json
from typing import Dict, List, Optional, Tuple
from loguru import logger
from openai import OpenAI

from config import (
    WORDS, COLORS, KEY_MAPPING,
    MESSAGES_PER_PRACTICE_TRIAL, SYSTEM_MESSAGE,
    MODEL, MAX_RETRIES, RETRY_DELAY,
    MESSAGES_PER_STIMULUS, MESSAGES_PER_SYSTEM,
    NUM_PARTICIPANTS,
    PRACTICE_TRIALS,
    TEST_TRIALS,
    RESPONSE_BUTTON_BLUE,
    RESPONSE_BUTTON_RED
)
from gpt4_old.image_handler import ImageHandler
from api_handler import APIHandlerFactory

class TrialManager:
    def __init__(self, client=None, use_dryrun=False):
        logger.debug("Initializing Trial Manager")
        self.image_handler = ImageHandler(overwrite_existing=True)
        self.api_handler = APIHandlerFactory.create_handler(client, use_dryrun)
        logger.debug(f"Using {'dry run' if use_dryrun else 'real'} API handler")
        self.key_mapping = KEY_MAPPING
    
    def _validate_message_count(self, messages, trial, practice):
        """Validate the current message count after adding new stimuli."""
        if practice:
            # For trial N, we should have:
            # - System message (1)
            # - Previous trials' messages ((N) * MESSAGES_PER_PRACTICE_TRIAL)
            # - Current stimulus message (1)
            expected_messages = MESSAGES_PER_SYSTEM + (trial * MESSAGES_PER_PRACTICE_TRIAL) + MESSAGES_PER_STIMULUS
            if len(messages) != expected_messages:
                error_msg = f"Context size error after adding stimuli: Expected {expected_messages} messages , but got {len(messages)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # After all practice trials, we should have:
            # - System message (1)
            # - All practice trials' messages (PRACTICE_TRIALS * MESSAGES_PER_PRACTICE_TRIAL)
            # - Current stimulus message (1)
            expected_messages = MESSAGES_PER_SYSTEM + (MESSAGES_PER_PRACTICE_TRIAL * PRACTICE_TRIALS) + MESSAGES_PER_STIMULUS
            if len(messages) != expected_messages:
                error_msg = f"Context size error after adding stimuli: Expected {expected_messages} messages (1 system + {PRACTICE_TRIALS} practice trials * {MESSAGES_PER_PRACTICE_TRIAL} + 1 stimulus), but got {len(messages)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _add_stimulus_messages(self, messages, base64_image):
        """Add the text and image stimulus messages to the conversation."""
        text_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"What color is this text? Respond with '{RESPONSE_BUTTON_BLUE}' for blue or '{RESPONSE_BUTTON_RED}' for red. ONLY respond with '{RESPONSE_BUTTON_BLUE}' or '{RESPONSE_BUTTON_RED}'! no other text or characters."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
        messages.append(text_message)

    def _log_trial_header(self):
        """Log the header for trial results table."""
        header = (
            "\n{:^6} | {:^10} | {:^8} | {:^6} | {:^6} | {:^8} | {:^8} | {:^3} | {:^7} | {:^7} | {:^7} | {:^10} | {:^8} | {:^10} | {:^10} | {:^10}"
            .format(
                "Part.", "Trial", "Type", "Word", "Color", "Response", 
                "Correct", "✓/✗", "In-Tok", "Out-Tok", "Tot-Tok", "LogProb",
                "Alt", "Alt-LogP", "Alt-Sum", "Cond."
            )
        )
        separator = "-" * len(header)
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

    def _log_trial_results(self, trial_result, participant_id, trial, practice, word, color, response, messages):
        """Log the results of a single trial."""
        total_trials = PRACTICE_TRIALS if practice else TEST_TRIALS
        
        trial_info = (
            "{:^6} | {:^10} | {:^8} | {:^6} | {:^6} | {:^8} | {:^8} | {:^3} | {:^7} | {:^7} | {:^7} | {:^10} | {:^8} | {:^10} | {:^10} | {:^10}"
            .format(
                f"{participant_id}/{NUM_PARTICIPANTS}",
                f"{trial + 1}/{total_trials}",
                "Practice" if practice else "Normal",
                word,
                trial_result['color'],
                response,
                trial_result['correct_response'],
                "✓" if trial_result['accuracy'] == 1 else "✗",
                trial_result['prompt_tokens'],
                trial_result['completion_tokens'],
                trial_result['total_tokens'],
                f"{trial_result['response_logprob']:.3f}" if isinstance(trial_result['response_logprob'], (float, int)) else 'N/A',
                trial_result['second_best_response'],
                f"{trial_result['second_best_logprob']:.3f}" if isinstance(trial_result['second_best_logprob'], (float, int)) else 'N/A',
                f"{trial_result['alternative_sum']:.3f}" if isinstance(trial_result['alternative_sum'], (float, int)) else 'N/A',
                trial_result['condition']
            )
        )
        logger.info(trial_info)
        
        # Print a blank line after each block of 10 trials
        if (trial + 1) % 10 == 0:
            logger.info("")

    def _handle_practice_feedback(self, trial_result, response, messages):
        """Handle practice trial feedback and get AI response."""
        if trial_result['accuracy'] == 1:
            feedback = f"Correct! The text color was '{trial_result['correct_response']}'."
        else:
            feedback = f"Incorrect. The text color was '{trial_result['correct_response']}'. Your response was '{response}'. Please try to be more accurate."
        
        # Add feedback message
        feedback_message = {
            "role": "user",
            "content": [{"type": "text", "text": feedback}]
        }
        messages.append(feedback_message)
        logger.debug(f"  Feedback: {feedback}")
        
        # Get AI response to feedback
        start_time = time.time()
        gpt_response = self.api_handler.get_response(messages)
        end_time = time.time()
        
        # Unpack response
        ai_response, reaction_time, total_time, retry_count, token_usage = gpt_response
        logger.debug(f"  Response: {ai_response}")
        
        # Add AI response to messages
        ai_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": ai_response["content"]}]
        }
        messages.append(ai_message)
        
        return feedback_message, ai_message

    def _validate_response(self, response: str) -> bool:
        """Validate that the response is either 'b' or 'm'.
        
        Args:
            response: The response to validate
            
        Returns:
            True if response is valid, False otherwise
        """
        is_valid = response.lower() in [RESPONSE_BUTTON_BLUE, RESPONSE_BUTTON_RED]
        if not is_valid:
            logger.warning(f"Invalid response received: {response}")
        return is_valid

    def _handle_invalid_response(self, messages: List[Dict], response: str, max_retries: int = 3) -> Dict:
        """Handle an invalid response by providing feedback and getting a new response.
        
        Args:
            messages: The current message history
            response: The invalid response
            max_retries: Maximum number of retries for invalid responses
            
        Returns:
            The corrected response
        """
        for attempt in range(max_retries):
            feedback = f"Invalid response: '{response}'. Please respond with ONLY '{RESPONSE_BUTTON_BLUE}' for blue or '{RESPONSE_BUTTON_RED}' for red."
            feedback_message = {
                "role": "user",
                "content": [{"type": "text", "text": feedback}]
            }
            messages.append(feedback_message)
            logger.debug(f"  Feedback attempt {attempt + 1}/{max_retries}: {feedback}")
            
            # Get corrected response
            start_time = time.time()
            gpt_response = self.api_handler.get_response(messages, max_tokens=1)  # Limit to 1 token
            end_time = time.time()
            
            # Unpack response
            ai_response, reaction_time, total_time, retry_count, token_usage = gpt_response
            logger.debug(f"  Corrected response: {ai_response}")
            
            # Validate the corrected response
            if self._validate_response(ai_response["content"]):
                # Add AI response to messages
                ai_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ai_response["content"]}]
                }
                messages.append(ai_message)
                return ai_response
            
            # If still invalid, remove the feedback and try again
            messages.pop()  # Remove feedback message
        
        # If all retries failed, use a default response
        logger.error(f"All {max_retries} retries failed for invalid response. Using default response.")
        default_response = RESPONSE_BUTTON_BLUE if self.key_mapping[color] == "b" else RESPONSE_BUTTON_RED
        ai_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": default_response}]
        }
        messages.append(ai_message)
        return ai_message

    def run_trial(self, trial, word, color, participant_id, messages, practice=False):
        """Run a single trial and return the results."""
        logger.debug(f"Starting {trial + 1} trial for participant {participant_id}")
        logger.debug(f"Initial message count: {len(messages)}")
        
        # Get image content for API request
        image_content = self.image_handler.get_stimuli_encoded(word, color)
        
        # Add stimulus messages
        self._add_stimulus_messages(messages, image_content)
        logger.debug(f"Message count after adding stimuli: {len(messages)}")

        # Validate message count after adding stimuli
        self._validate_message_count(messages, trial, practice)

        # Get GPT response
        response_message, retry_count, token_usage, logprobs = self.api_handler.get_response(messages, max_tokens=1)  # Limit to 1 token
        logger.debug(f"Message count after GPT response: {len(messages)}")
        
        # Create trial result with all parameters
        response = response_message["content"]
        
        # Validate response and get corrected response if needed
        if not self._validate_response(response):
            response_message = self._handle_invalid_response(messages, response)
            response = response_message["content"]
        
        correct_response = self.key_mapping[color]
        accuracy = 1 if response == correct_response else 0
        
        # Format color as a string
        color_name = 'blue' if color == (0, 0, 255) else 'red' if color == (255, 0, 0) else 'unknown'
        
        # Determine condition
        if word == "XXXX":
            condition = "neutral"
        elif (word == "BLUE" and color == (0, 0, 255)) or (word == "RED" and color == (255, 0, 0)):
            condition = "congruent"
        else:
            condition = "incongruent"
        
        # Extract token usage details
        prompt_tokens = token_usage.prompt_tokens
        completion_tokens = token_usage.completion_tokens
        total_tokens = token_usage.total_tokens
        
        # Extract logprob details and find opposite answer
        main_logprob = logprobs.get('logprob', 'N/A')
        top_logprobs = logprobs.get('top_logprobs', [])
        
        # Find the opposite answer in the logprobs
        opposite_answer = RESPONSE_BUTTON_RED if response == RESPONSE_BUTTON_BLUE else RESPONSE_BUTTON_BLUE
        opposite_logprob = None
        for alt in top_logprobs:
            if alt['token'].strip() == opposite_answer:
                opposite_logprob = alt['logprob']
                break
        
        # Calculate sum of likelihoods for all alternatives (excluding the main response)
        import math
        alternative_sum = 0.0
        for alt in top_logprobs:
            if alt['token'].strip() != response:  # Exclude the main response
                alternative_sum += math.exp(alt['logprob'])
        alternative_sum = math.log(alternative_sum) if alternative_sum > 0 else float('-inf')
        
        # Create trial result with formatted data
        trial_result = {
            "participant_id": participant_id,
            "trial": trial + 1,
            "word": word,
            "color": color_name,
            "response": response,
            "correct_response": correct_response,
            "accuracy": accuracy,
            "retry_count": retry_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_logprob": main_logprob,
            "second_best_response": opposite_answer,
            "second_best_logprob": opposite_logprob if opposite_logprob is not None else 'N/A',
            "alternative_sum": alternative_sum,
            "condition": condition
        }
        
        self._log_trial_results(trial_result, participant_id, trial, practice, word, color, response, messages)
        
        # Clean up messages after trial
        messages.pop()  # Remove stimulus message
        logger.debug(f"Message count after cleanup: {len(messages)}")
        
        return trial_result 