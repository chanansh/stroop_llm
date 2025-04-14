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
    TEST_TRIALS
)
from image_handler import ImageHandler
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
            expected_messages = MESSAGES_PER_SYSTEM + (MESSAGES_PER_PRACTICE_TRIAL * PRACTICE_TRIALS) +  MESSAGES_PER_STIMULUS
            if len(messages) != expected_messages:
                error_msg = f"Context size error after adding stimuli: Expected {expected_messages} messages (1 system + {PRACTICE_TRIALS} practice trials * {MESSAGES_PER_PRACTICE_TRIAL} + 1 stimulus), but got {len(messages)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _add_stimulus_messages(self, messages, base64_image):
        """Add the text and image stimulus messages to the conversation."""
        text_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this text? Respond with 'b' for blue or 'm' for red."},
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
            "\n{:^6} | {:^10} | {:^8} | {:^6} | {:^8} | {:^10} | {:^8} | {:^8} | {:^7} | {:^7} | {:^7} | {:^7} | {:^3} | {:^7} | {:^7} | {:^7}"
            .format("Part.", "Trial", "Type", "Word", "Color", "Type", "Response", "Correct", "RT(s)", "RTT(s)", "Total", "Ctx", "✓/✗", "In-Tok", "Out-Tok", "Tot-Tok")
        )
        separator = "-" * len(header)
        logger.info(header)
        logger.info(separator)

    def _log_trial_results(self, trial_result, participant_id, trial, practice, word, color, response, messages):
        """Log the results of a single trial."""
        total_trials = PRACTICE_TRIALS if practice else TEST_TRIALS
        is_mismatch = word != "XXXX" and ((word == "BLUE" and color == (255, 0, 0)) or (word == "RED" and color == (0, 0, 255)))
        
        trial_info = (
            "{:^6} | {:^10} | {:^8} | {:^6} | {:^8} | {:^10} | {:^8} | {:^8} | {:^7.3f} | {:^7.3f} | {:^7.3f} | {:^7} | {:^3} | {:^7} | {:^7} | {:^7}"
            .format(
                f"{participant_id}/{NUM_PARTICIPANTS}",
                f"{trial + 1}/{total_trials}",
                "Practice" if practice else "Normal",
                word,
                'red' if color == (255, 0, 0) else 'blue',
                'mismatch' if is_mismatch else 'match',
                response,
                trial_result['correct_response'],
                trial_result['api_time'],
                trial_result['rtt'],
                trial_result['total_time'],
                len(messages),
                "✓" if trial_result['accuracy'] == 1 else "✗",
                trial_result['input_tokens'],
                trial_result['output_tokens'],
                trial_result['total_tokens']
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
        is_valid = response.lower() in ['b', 'm']
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
            feedback = f"Invalid response: '{response}'. Please respond with ONLY 'b' for blue or 'm' for red."
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
        default_response = "b" if self.key_mapping[color] == "b" else "m"
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

        # Measure API RTT right before the trial
        rtt = self.api_handler.measure_rtt()

        # Get GPT response and measure time
        start_time = time.time()
        response_message, reaction_time, total_time, retry_count, token_usage = self.api_handler.get_response(messages, max_tokens=1)  # Limit to 1 token
        end_time = time.time()
        logger.debug(f"Message count after GPT response: {len(messages)}")
        
        # Create trial result with all parameters
        response = response_message["content"]
        
        # Validate response and get corrected response if needed
        if not self._validate_response(response):
            response_message = self._handle_invalid_response(messages, response)
            response = response_message["content"]
            # Update timing for the corrected response
            end_time = time.time()
            reaction_time = end_time - start_time
        
        correct_response = self.key_mapping[color]
        accuracy = 1 if response.lower() == correct_response else 0
        is_mismatch = word != "XXXX" and ((word == "BLUE" and color == (255, 0, 0)) or (word == "RED" and color == (0, 0, 255)))
        
        trial_result = {
            "participant_id": participant_id,
            "trial_number": trial + 1,
            "trial_type": "Practice" if practice else "Normal",
            "word": word,
            "color": 'red' if color == (255, 0, 0) else 'blue',
            "stimulus_type": 'mismatch' if is_mismatch else 'match',
            "gpt_response": response,
            "correct_response": correct_response,
            "accuracy": accuracy,
            "api_time": reaction_time,
            "rtt": rtt,
            "total_time": total_time + rtt,
            "retry_count": retry_count,
            "start_time": start_time,
            "end_time": end_time,
            "practice": practice,
            "context_size": len(messages),
            "input_tokens": token_usage.prompt_tokens,
            "output_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
        
        # Log trial results
        if trial == 0:
            self._log_trial_header()
        
        # Log results before adding response to messages
        self._log_trial_results(
            trial_result, participant_id, trial, practice, 
            word, color, response, messages
        )
        
        # Add the original AI response
        messages.append(response_message)
        
        # Handle practice feedback or cleanup
        if practice:
            feedback_message, ai_message = self._handle_practice_feedback(trial_result, response, messages)
        else:
            # For main trials, just remove the stimulus message
            messages.pop()  # Remove AI response
            messages.pop()  # Remove stimulus message
            logger.debug(f"Message count after removing stimulus: {len(messages)}")
        
        return trial_result 