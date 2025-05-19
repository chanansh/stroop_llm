import os
import json
import pprint
from glob import glob
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from plotly import express as px
import re

experiment_path = "results/multiword_experiment_2025-05-18_01-03-07"
logger.info(f"Experiment path: {experiment_path}")

def get_experiment_parameters(experiment_path: str):
    logger.info(f"Getting experiment parameters from {experiment_path}")
    with open(os.path.join(experiment_path, "experiment_parameters.json"), "r") as f:
        experiment_parameters = json.load(f)
    logger.info(f"Experiment parameters: {pprint.pformat(experiment_parameters)}")
    return experiment_parameters

def get_data(experiment_path: str, pattern: str = "trial_*.json"):
    experiment_file = os.path.join(experiment_path, "experiment_data.json")
    if os.path.exists(experiment_file):
        logger.info(f"Experiment data file {experiment_file} exists, loading it")
        with open(experiment_file, "r") as f:
            experiment_data = json.load(f)
        return experiment_data
    else:
        logger.info(f"Experiment data file {experiment_file} does not exist, reading trials from {experiment_path}")

    trials = []
    for file in glob(os.path.join(experiment_path, pattern)):
        with open(file, "r") as f:
            trials.append(json.load(f))
    logger.info(f"Read {len(trials)} trials")
    logger.info(f"Saving trials to {experiment_file}")
    with open(experiment_file, "w") as f:
        json.dump(trials, f)
    return trials

def get_correct_color(color:str, response:str, inverse_color_mapping:dict):
    color_name = inverse_color_mapping[color]
    return color_name.lower() == response.lower()

def get_correct_word(word:str, response:str):
    return word.lower() == response.lower()

def get_condition(word:str, color:str, mapping:dict):
    word_as_color = mapping[word]
    if word_as_color is None:
        return "neutral"
    elif word_as_color.lower() == color.lower():
        return "congruent"
    else:
        return "incongruent"

def get_number_of_words_from_path(path: str) -> int:
    """Extract number of words from directory name only."""
    dirname = os.path.basename(path)
    match = re.search(r'number_of_words_(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None

def analyze_single_experiment(experiment_path: str) -> pd.DataFrame:
    """Analyze a single experiment and return results DataFrame."""
    experiment_parameters = get_experiment_parameters(experiment_path)
    trials = get_data(experiment_path)
    color_mapping = experiment_parameters["color_mapping"]
    inverse_color_mapping = {v: k for k, v in color_mapping.items() if v is not None}
    
    # analyze number of responses per trial
    number_of_responses_per_trial = [len(t["response"]) for t in trials]
    expected_number_of_responses_per_trial = experiment_parameters["number_of_words_per_trial"]
    print(f"Expected number of responses per trial: {expected_number_of_responses_per_trial}, but got {np.mean(number_of_responses_per_trial)} +- {np.std(number_of_responses_per_trial)} (min: {np.min(number_of_responses_per_trial)}, max: {np.max(number_of_responses_per_trial)})")
    
    # distribution of number of responses per trial
    fig = px.histogram(number_of_responses_per_trial)
    fig.update_xaxes(title="Number of responses per trial")
    fig.update_yaxes(title="Count")
    fig.update_layout(showlegend=False)
    figures_path = os.path.join(experiment_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    fig.write_image(os.path.join(figures_path, "number_of_responses_per_trial_histogram.png"))
    
    results = []
    for trial in trials:
        for word_index, (word, color, response) in enumerate(zip(trial["word"], trial["color"], trial["response"])):
            response = response.lower().strip()
            condition = get_condition(word, color, color_mapping)
            correct_color = get_correct_color(color, response, inverse_color_mapping)
            correct_word = get_correct_word(word, response)
            results.append({
                "trial_id": trial["trial_id"],
                "word_index": word_index,
                "word": word,
                "color": color,
                "response": response,
                "condition": condition,
                "correct_color": correct_color,
                "correct_word": correct_word,
                "is_neutral": color_mapping[word] is None
            })
    
    df = pd.DataFrame(results)
    
    # probability to response with a name which is not a color
    df['valid_color'] = df['response'].isin([c.lower() for c in color_mapping.keys()])
    print(f"Probability to response with a name which is color: {np.mean(df['valid_color'])}")
    # examples for non-color responses
    logger.info("Non valid color responses")
    logger.info(df[~df['valid_color']].groupby(['color','response']).size().sort_values(ascending=False))
    # totally wrong responses
    logger.info("Totally wrong responses")
    df['totally_wrong'] = ~df['correct_color'] & ~df['correct_word']
    logger.info(df[df['totally_wrong']]['response'].value_counts())

    # Calculate mean and standard error for each condition
    summary = df.groupby('condition')['correct_color'].agg(['mean', 'count', 'std']).reset_index()
    summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
    
    # Create bar plot for accuracy by condition
    fig = px.bar(summary, x="condition", y="mean", color="condition",
                 error_y="stderr",
                 labels={"mean": "Accuracy"})
    fig.update_xaxes(title="Condition")
    fig.update_yaxes(title="Accuracy")
    fig.write_image(os.path.join(figures_path, "accuracy_by_condition_barplot.png"))
    
    # analyze errors - are they more likely to be the color of the word?
    df_errors = df[df['correct_color'] == False].copy()
    df_errors['naming_word'] = df_errors['response'] == df_errors['word']
    # Calculate mean and standard error for each condition
    summary_errors = df_errors.groupby("condition")['naming_word'].agg(['mean', 'count', 'std']).reset_index()
    summary_errors['stderr'] = summary_errors['std'] / np.sqrt(summary_errors['count'])
    
    # Create bar plot for naming word accuracy by condition
    fig = px.bar(summary_errors, x="condition", y="mean", color="condition",
                 error_y="stderr",
                 labels={"mean": "Accuracy"})
    fig.update_xaxes(title="Condition")
    fig.update_yaxes(title="Naming word accuracy")
    fig.write_image(os.path.join(figures_path, "naming_word_accuracy_by_condition_barplot.png"))
    
    return df

def extract_variable_from_dir(subdir: str):
    """Extracts the variable (number of words or spacing factor) and its value from the subdir name."""
    dirname = os.path.basename(subdir)
    # Try number_of_words
    match = re.search(r'number_of_words_(\d+)', dirname)
    if match:
        return 'n_words', int(match.group(1))
    # Try spacing_factor
    match = re.search(r'spacing_factor_([\d\.]+)', dirname)
    if match:
        return 'spacing_factor', float(match.group(1))
    return None, None

def analyze_meta_experiment(meta_experiment_path: str):
    """Analyze a meta-experiment containing multiple sub-experiments."""
    all_results = []
    variable_name = None
    variable_values = []
    
    # Process each subdirectory
    for subdir in sorted(glob(os.path.join(meta_experiment_path, '*'))):
        var_name, var_value = extract_variable_from_dir(subdir)
        if var_name is None:
            continue
        # Check for experiment_parameters.json
        param_file = os.path.join(subdir, "experiment_parameters.json")
        if not os.path.exists(param_file):
            logger.warning(f"Skipping {subdir}: experiment_parameters.json not found.")
            continue
        logger.info(f"Processing experiment with {var_name}={var_value}")
        df = analyze_single_experiment(subdir)
        df[var_name] = var_value
        all_results.append(df)
        variable_name = var_name
        variable_values.append(var_value)
    
    if not all_results:
        raise RuntimeError("No valid experiments found in meta-experiment directory.")
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate summary statistics
    summary = combined_df.groupby([variable_name, 'condition'])['correct_color'].agg(['mean', 'count', 'std']).reset_index()
    summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
    
    # Create line plot for accuracy by condition and variable
    fig = px.line(summary, x=variable_name, y="mean", color="condition",
                  error_y="stderr",
                  labels={"mean": "Accuracy", variable_name: variable_name.replace('_', ' ').title()},
                  markers=True)
    fig.update_layout(
        title=f"Accuracy by Condition and {variable_name.replace('_', ' ').title()}",
        xaxis_title=variable_name.replace('_', ' ').title(),
        yaxis_title="Accuracy",
        yaxis_range=[0, 1]
    )
    
    # Save the plot
    figures_path = os.path.join(meta_experiment_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    fig.write_image(os.path.join(figures_path, f"accuracy_by_condition_and_{variable_name}.png"))
    
    return combined_df, summary

def main():
    # For meta-experiment analysis
    meta_experiment_path = "results/multiword/meta/experiment_spacing_2025-05-18_01-19-35"
    logger.info(f"Analyzing meta-experiment: {meta_experiment_path}")
    combined_df, summary = analyze_meta_experiment(meta_experiment_path)
    
    # Print summary statistics
    print("\nMeta-experiment summary:")
    print(summary.to_string())

if __name__ == "__main__":
    main()