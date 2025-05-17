import os
import json
from openai import OpenAI
from loguru import logger
from gpt4_old.image_handler import ImageHandler
from src.config import (
    WORDS, COLORS, SYSTEM_MESSAGE,
    MODEL, MAX_RETRIES
)

def test_all_stimuli():
    """Test sending all stimuli to the LLM and verifying its responses."""
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize image handler
    image_handler = ImageHandler()
    
    # Get all combinations of words and colors
    for word in WORDS:
        for color in COLORS:
            # Get the image
            base64_image = image_handler.get_stimuli_encoded(word, color)
            
            # Create the message with both text and image
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe this image in detail. What do you see?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
            
            # Send the request
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[message],
                    max_tokens=100
                )
                
                # Print input/output pair
                print("\n" + "="*80)
                print(f"Input: Word='{word}', Color={color}")
                print("-"*80)
                print("LLM's description:")
                print(response.choices[0].message.content)
                print("="*80)
                
            except Exception as e:
                print(f"\nError processing word='{word}', color={color}: {str(e)}")
                continue

if __name__ == "__main__":
    test_all_stimuli() 