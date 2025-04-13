import time
import random
import os
from loguru import logger
from config import (
    MESSAGES_PER_STIMULUS, KEY_MAPPING,
    WORDS, COLORS, IMAGE_DIR,
    LOG_FILE, RESULTS_DIR,
    PRACTICE_TRIALS, STIMULI_COUNT,
    MESSAGES_PER_PRACTICE_TRIAL,
    NUM_PARTICIPANTS
)
from image_handler import ImageHandler
from api_handler import APIHandlerFactory
from utils import measure_ping
from typing import Tuple, Dict, Any

class TrialManager:
    def __init__(self, client=None, use_mock=False):
        logger.debug("Initializing Trial Manager")
        self.image_handler = ImageHandler()
        self.api_handler = APIHandlerFactory.create_handler(client, use_mock)
        logger.debug(f"Using {'mock' if use_mock else 'real'} API handler")
        self.key_mapping = KEY_MAPPING
    
    def _validate_message_count(self, messages, trial, practice):
        """Validate the current message count before adding new stimuli."""
        if practice:
            # For trial N, we should have:
            # - System message (1)
            # - Previous trials' messages ((N-1) * 5)
            expected_messages = 1 + trial * MESSAGES_PER_PRACTICE_TRIAL + MESSAGES_PER_STIMULUS
            if len(messages) != expected_messages:
                error_msg = f"Context size error before adding stimuli: Expected {expected_messages} messages , but got {len(messages)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # After all practice trials, we should have:
            # - System message (1)
            # - All practice trials' messages (PRACTICE_TRIALS * 5)
            expected_messages = 1 + (MESSAGES_PER_PRACTICE_TRIAL * PRACTICE_TRIALS) + MESSAGES_PER_STIMULUS
            if len(messages) != expected_messages:
                error_msg = f"Context size error before adding stimuli: Expected {expected_messages} messages (1 system + {PRACTICE_TRIALS} practice trials * 5), but got {len(messages)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def _add_stimulus_messages(self, messages, base64_image):
        """Add the text and image stimulus messages to the conversation."""
        text_message = {
            "role": "user",
            "content": "What color is this text? Respond with 'b' for blue or 'm' for red."
        }
        image_message = {
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        }
        messages.append(text_message)
        messages.append(image_message)

    def _log_trial_header(self):
        """Log the header for trial results table."""
        header = (
            "\n{:^8} | {:^12} | {:^8} | {:^8} | {:^10} | {:^10} | {:^9} | {:^10} | {:^8} | {:^8}"
            .format("Part.", "Trial", "Type", "Word", "Color", "Type", "Response", "Correct", "RT (s)", "Context")
        )
        separator = "-" * len(header)
        logger.info(header)
        logger.info(separator)

    def _log_trial_results(self, trial_result, participant_id, trial, practice, word, color, response, messages):
        """Log the results of a single trial."""
        total_trials = PRACTICE_TRIALS if practice else STIMULI_COUNT
        is_mismatch = word != "XXXX" and ((word == "BLUE" and color == (255, 0, 0)) or (word == "RED" and color == (0, 0, 255)))
        
        trial_info = (
            "{:^8} | {:^12} | {:^8} | {:^8s} | {:^8s} | {:^10s} | {:^10s} | {:^9s} | {:^8.3f} | {:^8}"
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
                len(messages)  # Add context length
            )
        )
        logger.info(trial_info)
        
        # Print a blank line after each block of 10 trials
        if (trial + 1) % 10 == 0:
            logger.info("")

    def _handle_practice_feedback(self, trial_result, response, messages):
        """Handle practice trial feedback and get AI response."""
        if trial_result['accuracy'] == 1:
            feedback = f"USER: Correct! The text color was '{trial_result['correct_response']}'."
        else:
            feedback = f"USER: Incorrect. The text color was '{trial_result['correct_response']}'. Your response was '{response}'. Please try to be more accurate."
        
        # Add feedback message
        feedback_message = {
            "role": "user",
            "content": feedback
        }
        messages.append(feedback_message)
        logger.info(f"  Feedback: {feedback}")
        
        # Get AI response to feedback
        start_time = time.time()
        gpt_response = self.api_handler.get_response(messages)
        end_time = time.time()
        
        # Unpack response
        ai_response, reaction_time, total_time, retry_count = gpt_response
        logger.info(f"  Response: {ai_response}")
        
        # Add AI response to messages
        ai_message = {
            "role": "assistant",
            "content": ai_response
        }
        messages.append(ai_message)
        
        return feedback_message, ai_message

    def run_trial(self, trial, word, color, participant_id, messages, practice=False):
        """Run a single trial and return the results."""
        logger.debug(f"Starting {trial + 1} trial for participant {participant_id}")
        logger.debug(f"Initial message count: {len(messages)}")
        
        # Get stimulus image path and encode it
        img_path = self.image_handler.get_stimulus_path(word, color)
        base64_image = self.image_handler.encode_image(img_path)
        
        # Measure ping time right before the trial
        ping_time = measure_ping(count=1)
               
        # Add stimulus messages
        self._add_stimulus_messages(messages, base64_image)
        logger.debug(f"Message count after adding stimuli: {len(messages)}")
        # Validate message count before adding stimuli
        self._validate_message_count(messages, trial, practice)

        # Get GPT response and measure time
        start_time = time.time()
        response_message, reaction_time, total_time, retry_count = self.api_handler.get_response(messages)
        end_time = time.time()
        logger.debug(f"Message count after GPT response: {len(messages)}")
        
        # Create trial result
        response = response_message["content"]
        correct_response = self.key_mapping[color]
        accuracy = 1 if response.lower() == correct_response else 0
        
        trial_result = {
            "participant_id": participant_id,
            "trial_number": trial + 1,
            "word": word,
            "color": color,
            "gpt_response": response,
            "correct_response": correct_response,
            "accuracy": accuracy,
            "api_time": reaction_time,
            "ping_time": ping_time,
            "total_time": total_time + ping_time,
            "retry_count": retry_count,
            "start_time": start_time,
            "end_time": end_time,
            "practice": practice
        }
        
        # Log trial results
        if trial == 0:
            self._log_trial_header()
        
        # Log results before adding response to messages
        self._log_trial_results(
            trial_result, participant_id, trial, practice, 
            word, color, response, messages
        )
        
        # Add AI's response to messages after logging
        messages.append(response_message)
        
        # Handle practice feedback
        if practice:
            feedback_message, ai_message = self._handle_practice_feedback(trial_result, response, messages)
        else:
            # For main trials, remove the stimulus messages
            messages.pop()  # Remove AI response
            messages.pop()  # Remove image message
            messages.pop()  # Remove text message
            logger.debug(f"Message count after removing stimuli: {len(messages)}")
        
        return trial_result 