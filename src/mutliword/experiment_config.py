from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class ExperimentConfig:
    complex_color_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Color words
        "Purple": "purple",
        "Green": "green", 
        "Brown": "brown",
        "Blue": "blue",
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
    simple_color_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Color words
        "Blue": "blue",
        "Red": "red",
        # Neutral word
        "XXXX": None
    })
    use_complex_mapping: bool = False  # Flag to choose between complex and simple mapping
    color_mapping: Dict[str, str] = field(init=False)  # Will be set in __post_init__
    neutral_word_probability: float = 0.2  # 20% chance of a word being neutral
    number_of_trials: int = 100
    number_of_words_per_trial: int = 1
    words_per_row: int = 1
    char_spacing: float = 1
    line_spacing: float = 0.5  # Increase for a bigger gap between rows
    font_size: int = 30
    font_name: str = "Arial.ttf"
    background_color: str = "white"
    alignment: str = "center"
    model: str = "gpt-4o-mini" # "gpt-4o-mini" or "o4-mini", gpt-4o
    experiment_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output_path: str = field(default_factory=lambda: f"results/multiword/experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    max_retries: int = 3
    max_api_retries: int = 5  # Maximum number of retries for API calls
    initial_retry_delay: float = 1.0  # Initial delay for retries in seconds
    seed: int = 42
    state_expected_length: bool = True

    def __post_init__(self):
        self.color_mapping = self.complex_color_mapping if self.use_complex_mapping else self.simple_color_mapping

    @property
    def prompt(self) -> str:
        length_info = f" (expecting {self.number_of_words_per_trial} color{'s' if self.number_of_words_per_trial > 1 else ''})" if self.state_expected_length else ""
        if self.number_of_words_per_trial == 1:
            return f"name the word's ink color, ignore the word meaning. Return only a valid json of a flat list of the word ink color{length_info}."
        elif self.number_of_words_per_trial <= self.words_per_row:
            return f"name the words' ink colors, ignore the word meaning, left to right. Return only a valid json of a flat list of the words' ink colors{length_info}."
        else:
            return f"name the words' ink colors, ignore the word meaning, line by line from left to right. Return only a valid json of a flat list of the ink colors{length_info}."