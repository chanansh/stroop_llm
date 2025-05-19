import pytest
from gpt4_matrix.multiword_experiment import get_words_and_colors, ExperimentConfig
from collections import Counter

def test_get_words_and_colors_distribution():
    # Create config
    config = ExperimentConfig()
    
    # Run multiple trials to get a good sample
    num_trials = 1000
    total_congruent = 0
    total_incongruent = 0
    
    for _ in range(num_trials):
        words, colors = get_words_and_colors(config)
        for word, color in zip(words, colors):
            if color == config.color_mapping[word]:
                total_congruent += 1
            else:
                total_incongruent += 1
    
    # Calculate percentages
    total = total_congruent + total_incongruent
    congruent_percentage = (total_congruent / total) * 100
    incongruent_percentage = (total_incongruent / total) * 100
    
    # Allow for some deviation (e.g., ±5%)
    assert 45 <= congruent_percentage <= 55, f"Congruent percentage {congruent_percentage}% is outside expected range"
    assert 45 <= incongruent_percentage <= 55, f"Incongruent percentage {incongruent_percentage}% is outside expected range"
    
    # Test that we get the correct number of words
    words, colors = get_words_and_colors(config)
    assert len(words) == config.number_of_words_per_trial
    assert len(colors) == config.number_of_words_per_trial
    
    # Test that all colors are valid
    for color in colors:
        assert color in config.color_mapping.values() 