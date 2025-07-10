# measure the effect of word spacing and number of words on the stroop effect

from datetime import datetime
import os
import pprint

import numpy as np
from mutliword.experiment_config import ExperimentConfig
from mutliword.multiword_experiment import run_experiment
from loguru import logger
from copy import deepcopy

meta_folder = "results/multiword/meta"

def test_spacing(default_experiment: ExperimentConfig, 
                 number_of_words_per_trial: int, 
                 words_per_row: int,
                 spacing_factor_list: list[float]):
    logger.info("Testing spacing" + "\n" + "="*80)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta_experiemnt_folder = os.path.join(meta_folder, "experiment_spacing_" + datetime_str)
    for spacing_factor in spacing_factor_list:
        logger.info(f"Running experiment with spacing factor {spacing_factor:.1f}")
        experiment = deepcopy(default_experiment)
        experiment.char_spacing = experiment.char_spacing * spacing_factor
        experiment.line_spacing = experiment.line_spacing * spacing_factor
        experiment.number_of_words_per_trial = number_of_words_per_trial
        experiment.words_per_row = words_per_row
        experiment.output_path = os.path.join(meta_experiemnt_folder, f"spacing_factor_{spacing_factor:.1f}")
        run_experiment(experiment)

def test_number_of_words(default_experiment: ExperimentConfig, 
                        number_of_words_per_trial_list: list[int]):
    logger.info("Testing number of words" + "\n" + "="*80)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta_experiemnt_folder = os.path.join(meta_folder, "experiment_number_of_words_" + datetime_str)
    for number_of_words in number_of_words_per_trial_list:
        words_per_row = int(np.sqrt(number_of_words))
        logger.info(f"Running experiment with number of words {number_of_words:d}\n{'-'*80}")
        experiment = deepcopy(default_experiment)
        experiment.number_of_words_per_trial = number_of_words
        experiment.words_per_row = words_per_row
        logger.info(pprint.pformat(experiment))
        experiment.output_path = os.path.join(meta_experiemnt_folder, f"number_of_words_{number_of_words:d}")
        run_experiment(experiment)

if __name__ == "__main__":
    default_experiment = ExperimentConfig()
    number_of_words_per_trial_list = [x**2 for x in range(1, 7)]
    test_number_of_words(default_experiment, number_of_words_per_trial_list)
    spacing_factor_list = [0.5, 1, 2, 4, 8]
    test_spacing(default_experiment, 
                 number_of_words_per_trial=16, 
                 words_per_row=4, 
                 spacing_factor_list=spacing_factor_list)