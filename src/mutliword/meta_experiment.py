# measure the effect of word spacing and number of words on the stroop effect

from datetime import datetime
import os
from mutliword.experiment_config import ExperimentConfig
from mutliword.multiword_experiment import run_experiment
from loguru import logger
from copy import deepcopy

def test_spacing(default_experiment: ExperimentConfig):
    logger.info("Testing spacing" + "\n" + "="*80)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta_experiemnt_folder = "results/multi_word/meta/experiment_spacing_" + datetime_str
    spacing_factor_list = [0.5, 1, 2, 4]
    for spacing_factor in spacing_factor_list:
        logger.info(f"Running experiment with spacing factor {spacing_factor:.1f}")
        experiment = deepcopy(default_experiment)
        experiment.char_spacing = experiment.char_spacing * spacing_factor
        experiment.line_spacing = experiment.line_spacing * spacing_factor
        experiment.output_path = os.path.join(meta_experiemnt_folder, f"spacing_factor_{spacing_factor:.1f}")
        run_experiment(experiment)

def test_number_of_words(default_experiment: ExperimentConfig):
    logger.info("Testing number of words" + "\n" + "="*80)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta_experiemnt_folder = "results/multi_word/meta/experiment_number_of_words_" + datetime_str
    number_of_words_per_trial_list = [3, 6, 12, 24]
    for number_of_words in number_of_words_per_trial_list:
        logger.info(f"Running experiment with number of words {number_of_words:d}")
        experiment = deepcopy(default_experiment)
        experiment.number_of_words_per_trial = number_of_words
        experiment.output_path = os.path.join(meta_experiemnt_folder, f"number_of_words_{number_of_words:d}")
        run_experiment(experiment)

if __name__ == "__main__":
    default_experiment = ExperimentConfig()
    test_spacing(default_experiment)
    test_number_of_words(default_experiment)
