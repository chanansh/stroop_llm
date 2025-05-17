from datetime import datetime
import os
import torch
from PIL import Image
from typing import Dict, List, Tuple
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from config import WORDS, COLORS
from gpt4_old.image_handler import ImageHandler
from transformers import CLIPProcessor, CLIPModel

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
        
        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set device - support MPS on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Define class sets for analysis
        self.experiments = {
            "meaning_only": [
                'The word "BLUE"',
                'The word "RED"'
            ],
            "color_only": [
                "A word colored in blue",
                "A word colored in red"
            ],
            "combined": [
                'The word "BLUE"',
                'The word "RED"',
                'A word colored in blue',
                'A word colored in red'
            ]
        }
        
    def _process_image(self, word: str, color: Tuple[int, int, int], candidate_labels: List[str]) -> Dict:
        """Process an image with CLIP model.
        
        Args:
            word: The word in the image
            color: RGB color tuple for the word
            candidate_labels: List of labels to classify against
            
        Returns:
            Dictionary containing predictions
        """
        try:
            logger.debug(f"Processing image for word '{word}' with color {color}")
            logger.debug(f"Candidate labels: {candidate_labels}")
            
            # Get image path from ImageHandler and open with PIL
            image_path = self.image_handler.get_stimulus_path(word, color)
            image = Image.open(image_path)
            
            # Process image and text separately
            image_inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            text_inputs = self.processor(
                text=candidate_labels,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to same device as model
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                text_features = self.model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarity[0]
            
            # Convert to dictionary
            result = [{"label": label, "score": score.item()} for label, score in zip(candidate_labels, probs)]
            
            logger.debug(f"Model predictions: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Model processing failed: {str(e)}")
            raise RuntimeError(f"Failed to process image with CLIP model: {str(e)}")
            
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
            for set_name, classes in self.experiments.items():
                try:
                    logger.debug(f"Processing experiment: {set_name}")
                    # Get CLIP predictions
                    model_response = self._process_image(word, color, classes)
                    
                    # Convert to probability dictionary
                    probs = {pred["label"]: pred["score"] for pred in model_response}
                    results[set_name] = probs
                    
                    logger.info(f"Results for {set_name}:")
                    for label, score in probs.items():
                        logger.info(f"  {label}: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing experiment {set_name}: {str(e)}")
                    raise RuntimeError(f"Failed to process experiment {set_name}") from e
                    
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
                            "experiment": set_name,
                            "prompt": class_name,
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
    
    # Print summary
    logger.info("\nSummary of results:")
    summary = results.groupby(['word', 'color', 'experiment']).apply(
        lambda x: x.loc[x['probability'].idxmax(), ['prompt', 'probability']]
    ).reset_index()
    logger.info(summary.to_string())

if __name__ == "__main__":
    main() 