from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class ExperimentConfig:
    color_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Color words
        "Purple": "darkviolet",
        "Green": "lime",
        "Brown": "peru",
        "Blue": "royalblue",
        "Red": "red",
        # Neutral words
        "XXXX": None,
        "Dog": None,
        "Cat": None,
        "Tree": None,
        "Book": None,
        "House": None,
        "Car": None,
        "Bird": None,
        "Fish": None,
        "Star": None
    })
    neutral_word_probability: float = 0.2  # 20% chance of a word being neutral
    number_of_trials: int = 100
    number_of_words_per_trial: int = 12
    words_per_row: int = 3
    char_spacing: float = 1
    line_spacing: float = 0.5  # Increase for a bigger gap between rows
    font_size: int = 30
    font_name: str = "Arial.ttf"
    background_color: str = "white"
    alignment: str = "center"
    prompt: str = (
        "name the words' ink colors, ignore the word meaning, line by line from left to right. Return only a valid json of a flat list of the ink colors."
    )
    model: str = "gpt-4o-mini" # "gpt-4o-mini" or "o4-mini"
    experiment_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output_path: str = field(default_factory=lambda: f"results/multi_word/experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/data")
    max_retries: int = 3
    max_api_retries: int = 5  # Maximum number of retries for API calls
    initial_retry_delay: float = 1.0  # Initial delay for retries in seconds
    seed: int = 42