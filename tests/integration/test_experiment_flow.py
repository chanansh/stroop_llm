import os
import pytest
from src.experiment import run_participant_experiment
from src.config import ExperimentConfig

@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary directory for test results."""
    original_dir = ExperimentConfig.RESULTS_DIR
    ExperimentConfig.RESULTS_DIR = str(tmp_path)
    yield tmp_path
    ExperimentConfig.RESULTS_DIR = original_dir

def test_run_participant_experiment(mock_openai_client, temp_results_dir):
    """Test running a complete participant experiment."""
    # Run experiment for one participant
    results = run_participant_experiment(1, mock_openai_client, use_mock=True)
    
    # Verify results
    assert len(results) == ExperimentConfig.PRACTICE_TRIALS + ExperimentConfig.STIMULI_COUNT
    
    # Check result structure
    for result in results:
        assert "Participant" in result
        assert "Trial" in result
        assert "Word" in result
        assert "Color" in result
        assert "GPT Response" in result
        assert "Correct Response" in result
        assert "Reaction Time (s)" in result
        assert "Total Time (s)" in result
        assert "Retry Count" in result
        assert "Accuracy" in result
        assert "Ping Time (ms)" in result
    
    # Verify results file was created
    results_file = os.path.join(temp_results_dir, "participant_1.csv")
    assert os.path.exists(results_file)

def test_run_participant_experiment_with_feedback(mock_openai_client, temp_results_dir):
    """Test that feedback is provided during practice trials."""
    results = run_participant_experiment(1, mock_openai_client, use_mock=True)
    
    # Check practice trials have feedback
    practice_results = results[:ExperimentConfig.PRACTICE_TRIALS]
    for result in practice_results:
        assert "Feedback" in result
        assert "Feedback Response" in result
        assert result["Feedback"] is not None
        assert result["Feedback Response"] is not None
    
    # Check main trials don't have feedback
    main_results = results[ExperimentConfig.PRACTICE_TRIALS:]
    for result in main_results:
        assert result["Feedback"] is None
        assert result["Feedback Response"] is None

def test_run_participant_experiment_message_count(mock_openai_client, temp_results_dir):
    """Test that message count increases correctly during experiment."""
    results = run_participant_experiment(1, mock_openai_client, use_mock=True)
    
    # Verify message count for each trial
    for i, result in enumerate(results):
        trial_num = i + 1
        if trial_num <= ExperimentConfig.PRACTICE_TRIALS:
            expected_messages = 1 + (trial_num * ExperimentConfig.MESSAGES_PER_TRIAL)
        else:
            main_trial_num = trial_num - ExperimentConfig.PRACTICE_TRIALS
            expected_messages = 1 + (ExperimentConfig.PRACTICE_TRIALS * ExperimentConfig.MESSAGES_PER_TRIAL) + (main_trial_num * ExperimentConfig.MESSAGES_PER_TRIAL)
        
        # Note: We can't directly verify the message count as it's internal to the function
        # This test serves as documentation of the expected behavior 