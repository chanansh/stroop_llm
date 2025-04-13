# Stroop LLM Experiment

A Python-based system for running Stroop effect experiments with Large Language Models (LLMs). The system tests how well LLMs can identify the color of text while ignoring its semantic meaning.

## Overview

The Stroop effect is a psychological phenomenon where naming the color of a word is more difficult when the word itself names a different color. This experiment adapts the classic Stroop test for LLMs, measuring their ability to:
- Identify text colors while ignoring word meanings
- Learn from feedback during practice trials
- Maintain consistent performance across trials

## Features

- **Stimulus Generation**: Creates colored text images for the experiment
- **Practice Trials**: Allows the model to learn from feedback
- **Main Trials**: Measures model performance without feedback
- **Detailed Logging**: Tracks all experiment events and results
- **Mock Mode**: Enables testing without API calls
- **Network Latency Measurement**: Accounts for network delays in reaction times
- **Command Line Interface**: Easy-to-use CLI for running experiments

## Project Structure

```
src/
├── base_api_handler.py    # Base class for API handlers
├── api_handler.py         # Real OpenAI API handler
├── mock_api_handler.py    # Mock API handler for testing
├── config.py             # Experiment configuration
├── experiment.py         # Main experiment orchestration
├── image_handler.py      # Stimulus image generation
├── trial_manager.py      # Individual trial management
├── utils.py             # Utility functions
├── cli.py              # Command line interface
└── test_experiment.py   # Test script
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
poetry install
```

## Configuration

Edit `src/config.py` to customize:
- Number of trials
- Stimulus words and colors
- API settings
- File paths
- Experiment parameters

## Usage

### Command Line Interface

The experiment can be run using the CLI:

```bash
# Run with default settings
poetry run python src/cli.py

# Run in mock mode
poetry run python src/cli.py --mock

# Run with custom number of participants
poetry run python src/cli.py --participants 5

# Run with custom log file
poetry run python src/cli.py --log-file custom.log

# Show help
poetry run python src/cli.py --help
```

### Results

Results are saved in the `results` directory:
- Individual participant results: `participant_{id}.csv`
- Combined results: `all_participants.csv`

## Development

### Testing

```bash
# Run tests
poetry run python src/test_experiment.py
```

### Adding New Features

1. Follow the existing architecture:
   - Keep configuration in `config.py`
   - Use the API handler abstraction
   - Maintain detailed logging
   - Follow the experiment flow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License 