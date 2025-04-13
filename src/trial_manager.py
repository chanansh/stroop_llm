import time
import random
from loguru import logger
from .config import ExperimentConfig
from .image_handler import ImageHandler
from .api_handler import APIHandlerFactory

class TrialManager:
    def __init__(self, client=None, use_mock=False):
        logger.info("Initializing Trial Manager")
        self.config = ExperimentConfig
        self.image_handler = ImageHandler()
        self.api_handler = APIHandlerFactory.create_handler(client, use_mock)
        logger.debug(f"Using {'mock' if use_mock else 'real'} API handler")
        self.key_mapping = {
            (0, 0, 255): "b",  # Blue color -> 'b'
            (255, 0, 0): "m"  # Red color -> 'm'
        }
    
    def run_trial(self, trial, word, color, participant_id, messages, practice=False):
        """Run a single trial."""
        trial_type = "practice" if practice else "main"
        logger.info(f"Starting {trial_type} trial {trial+1} for participant {participant_id}")
        logger.debug(f"Stimulus: Word='{word}', Color={color}")
        
        # Get image and prepare content
        img_filename = self.image_handler.get_image_filename(word, color)
        logger.debug(f"Using image: {img_filename}")
        start_time = time.time()
        
        content = self.image_handler.prepare_image_content(img_filename)
        messages.append({"role": "user", "content": content})
        logger.debug(f"Message context length: {len(messages)}")
        
        # Get and validate response
        gpt_response = self.api_handler.get_response(messages)
        logger.debug(f"Raw response: {gpt_response}")
        
        if not self.api_handler.validate_response(gpt_response):
            logger.warning(f"Invalid response received: {gpt_response}")
            messages.append({"role": "user", "content": "Invalid response. Please respond with 'b' or 'm'."})
            gpt_response = self.api_handler.get_response(messages)
            logger.debug(f"Second attempt response: {gpt_response}")
        
        # Process trial results
        trial_result = self._process_trial_results(
            start_time=start_time,
            trial=trial,
            word=word,
            color=color,
            participant_id=participant_id,
            gpt_response=gpt_response,
            messages=messages,
            practice=practice
        )
        
        logger.info(f"Trial {trial+1} completed. Accuracy: {trial_result['Accuracy']}, RT: {trial_result['Reaction Time (s)']:.3f}s")
        return trial_result
    
    def _process_trial_results(self, start_time, trial, word, color, participant_id, 
                             gpt_response, messages, practice):
        """Process and return trial results."""
        end_time = time.time()
        rt = end_time - start_time
        
        # Determine accuracy
        correct_response = self.key_mapping[color]
        accuracy = 1 if gpt_response == correct_response else 0
        
        if accuracy == 0:
            logger.warning(f"Incorrect response: Expected {correct_response}, got {gpt_response}")
        
        # Handle feedback if practice
        feedback = None
        feedback_response = None
        if practice:
            feedback = "Correct!" if accuracy else f"Incorrect. The correct response is '{correct_response}'"
            messages.append({"role": "user", "content": feedback})
            logger.debug(f"Feedback provided: {feedback}")
            
            feedback_response = self.api_handler.get_feedback_response(messages)
            logger.debug(f"Feedback response: {feedback_response}")
        
        # Return trial results
        return {
            "Participant": participant_id,
            "Trial": trial + 1,
            "Word": word,
            "Color": color,
            "GPT Response": gpt_response,
            "Correct Response": correct_response,
            "Reaction Time (s)": rt,
            "Accuracy": accuracy,
            "Feedback": feedback,
            "Feedback Response": feedback_response
        } 