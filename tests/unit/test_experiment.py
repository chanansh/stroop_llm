import pytest
from src.experiment import generate_trial_sequence
from src.config import ExperimentConfig

def test_generate_trial_sequence_length():
    """Test that generate_trial_sequence returns correct number of trials."""
    num_trials = 10
    sequence = generate_trial_sequence(num_trials)
    assert len(sequence) == num_trials

def test_generate_trial_sequence_valid_stimuli():
    """Test that all generated trials use valid stimuli."""
    num_trials = 100  # Large number to test randomness
    sequence = generate_trial_sequence(num_trials)
    
    for word, color in sequence:
        assert (word, color) in ExperimentConfig.STIMULI

def test_generate_trial_sequence_randomness():
    """Test that generate_trial_sequence produces different sequences."""
    num_trials = 10
    sequence1 = generate_trial_sequence(num_trials)
    sequence2 = generate_trial_sequence(num_trials)
    
    # It's possible (though unlikely) for two random sequences to be identical
    # So we'll check that at least some elements are different
    assert any(s1 != s2 for s1, s2 in zip(sequence1, sequence2))

def test_generate_trial_sequence_zero_trials():
    """Test that generate_trial_sequence handles zero trials."""
    sequence = generate_trial_sequence(0)
    assert len(sequence) == 0

def test_generate_trial_sequence_negative_trials():
    """Test that generate_trial_sequence handles negative trials."""
    with pytest.raises(ValueError):
        generate_trial_sequence(-1) 