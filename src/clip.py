from datetime import datetime
import os
import base64
import requests
import torch
from PIL import Image
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from config import WORDS, COLORS
from image_handler import ImageHandler

# Load environment variables
load_dotenv()

class CLIPAnalyzer:
    """Analyzer for CLIP model predictions on Stroop stimuli."""
    
    def __init__(self, image_handler: ImageHandler):
        """Initialize CLIP analyzer with image handler.
        
        Args:
            image_handler: Handler for stimulus images
        """
        self.image_handler = image_handler
        
        # Get API key from environment
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
            
        # Set up API endpoint and headers
        self.api_url = "https://router.huggingface.co/hf-inference/models/openai/clip-vit-base-patch32"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Define class sets for analysis
        self.class_sets = {
            "meaning_only": [
                "the text's meaning is red",
                "the text's meaning is blue"
            ],
            "color_only": [
                "the text's font color is red",
                "the text's font color is blue"
            ],
            "combined": [
                "the text's meaning is red and the text's font color is red",
                "the text's meaning is red and the text's font color is blue",
                "the text's meaning is blue and the text's font color is red",
                "the text's meaning is blue and the text's font color is blue"
            ],
            "combined_color_first": [
                "the text's font color is red and the text's meaning is red",
                "the text's font color is red and the text's meaning is blue",
                "the text's font color is blue and the text's meaning is red",
                "the text's font color is blue and the text's meaning is blue"
            ]
        }
        
    def _query_api(self, word: str, color: Tuple[int, int, int], candidate_labels: List[str]) -> Dict:
        """Query the Hugging Face API for CLIP predictions.
        
        Args:
            word: The word in the image
            color: RGB color tuple for the word
            candidate_labels: List of labels to classify against
            
        Returns:
            API response containing predictions
        """
        try:
            logger.debug(f"Querying API for word '{word}' with color {color}")
            logger.debug(f"Candidate labels: {candidate_labels}")
            
            # Get encoded image directly from ImageHandler
            encoded_image = self.image_handler.get_stimuli_encoded(word, color)
            
            payload = {
                "parameters": {"candidate_labels": candidate_labels},
                "inputs": encoded_image
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"API response: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise RuntimeError(f"Failed to query CLIP API: {str(e)}")
            
    def analyze_image(self, word: str, color: Tuple[int, int, int]) -> Dict[str, Dict[str, float]]:
        """Analyze a single image with all class sets using CLIP.
        
        Args:
            word: The word in the image
            color: RGB color tuple for the word
            
        Returns:
            Dictionary containing probabilities for each class set
            
        Raises:
            RuntimeError: If CLIP processing fails
        """
        try:
            logger.info(f"Starting analysis for word '{word}' with color {color}")
                
            results = {}
            for set_name, classes in self.class_sets.items():
                try:
                    logger.debug(f"Processing class set: {set_name}")
                    # Get CLIP predictions from API
                    api_response = self._query_api(word, color, classes)
                    
                    # Convert to probability dictionary
                    probs = {pred["label"]: pred["score"] for pred in api_response}
                    results[set_name] = probs
                    
                    logger.info(f"Results for {set_name}:")
                    for label, score in probs.items():
                        logger.info(f"  {label}: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing class set {set_name}: {str(e)}")
                    raise RuntimeError(f"Failed to process class set {set_name}") from e
                    
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing image for word '{word}' with color {color}: {str(e)}")
            raise
    
    def analyze_all_images(self) -> pd.DataFrame:
        """Analyze all images in the stimuli_images directory.
        
        Returns:
            DataFrame containing results for all images
        """
        logger.info("Starting analysis of all images")
        results = []
        
        # Analyze each word-color combination
        for word in WORDS:
            for color in COLORS:
                logger.info(f"\nAnalyzing word: {word}, color: {color}")
                
                # Get CLIP results
                clip_results = self.analyze_image(word, color)
                
                # Store results
                for set_name, probs in clip_results.items():
                    for class_name, prob in probs.items():
                        results.append({
                            "word": word,
                            "color": "red" if color == (255, 0, 0) else "blue",
                            "class_set": set_name,
                            "class_name": class_name,
                            "probability": prob
                        })
        
        logger.info(f"\nAnalysis complete. Processed {len(results)} results")
        return pd.DataFrame(results)

def main():
    logger.info("Initializing CLIP analyzer")
    # Create analyzer and run analysis
    analyzer = CLIPAnalyzer(ImageHandler())
    results = analyzer.analyze_all_images()
    
    # Save results
    output_dir = "results"
    filename = f"clip_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_path = os.path.join(output_dir, filename)  
    results.to_csv(full_path, index=False)
    logger.info(f"Results saved to {full_path}")
    logger.info("\nSummary of results:")
    logger.info(results.groupby(['word', 'color', 'class_set'])['probability'].mean().to_string())

if __name__ == "__main__":
    main() 