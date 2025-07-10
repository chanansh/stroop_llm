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
import argparse
import random
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from PIL import Image
from typing import List, Dict, Tuple
from Levenshtein import distance as levenshtein_distance
from string2string.alignment import NeedlemanWunsch
from string2string.misc import Tokenizer

legit_color_confusion = {"orange": "brown"}
def get_experiment_parameters(experiment_path: str):
    logger.info(f"Getting experiment parameters from {experiment_path}")
    with open(os.path.join(experiment_path, "experiment_parameters.json"), "r") as f:
        experiment_parameters = json.load(f)
    logger.info(f"Experiment parameters: {pprint.pformat(experiment_parameters)}")
    return experiment_parameters

def get_data(experiment_path: str, pattern: str = "trial_*.json"):
    """Get experiment data from either a single experiment or a meta-experiment."""
    # First try to find experiment_data.json
    experiment_file = os.path.join(experiment_path, "experiment_data.json")
    if os.path.exists(experiment_file):
        logger.info(f"Experiment data file {experiment_file} exists, loading it")
        with open(experiment_file, "r") as f:
            experiment_data = json.load(f)
        return experiment_data
    
    # If no experiment_data.json, look for trial files
    logger.info(f"Experiment data file {experiment_file} does not exist, reading trials from {experiment_path}")
    
    # Check if this is a single experiment or meta-experiment
    if os.path.isdir(os.path.join(experiment_path, "number_of_words_1")):
        # This is a meta-experiment, get data from subdirectories
        trials = []
        for subdir in glob(os.path.join(experiment_path, "number_of_words_*")):
            for file in glob(os.path.join(subdir, pattern)):
                with open(file, "r") as f:
                    trials.append(json.load(f))
    else:
        # This is a single experiment, get data directly from the directory
        trials = []
        for file in glob(os.path.join(experiment_path, pattern)):
            with open(file, "r") as f:
                trials.append(json.load(f))
    
    logger.info(f"Read {len(trials)} trials")
    
    # Save the trials to experiment_data.json for future use
    logger.info(f"Saving trials to {experiment_file}")
    with open(experiment_file, "w") as f:
        json.dump(trials, f)
    
    return trials

def get_expected_color_name(color:str, inverse_color_mapping:dict):
    color_name = inverse_color_mapping[color]
    return color_name.lower()

def is_correct_color(color_name:str, response:str, legit_color_confusion:dict):
    response_name = get_response_color_name(response, legit_color_confusion)
    return color_name == response_name

def get_response_color_name(response:str, legit_color_confusion:dict):
    response = response.lower()
    return legit_color_confusion.get(response, response)


def is_valid_color(response:str, inverse_color_mapping:dict, legit_color_confusion:dict):
    response_color_name = get_response_color_name(response, legit_color_confusion)
    return response_color_name in [c.lower() for c in inverse_color_mapping.values()]

def is_correct_word(word:str, response:str):
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

def create_example_stimuli_figure(df: pd.DataFrame, experiment_path: str):
    """Create a figure showing example stimuli from each condition using actual trial screenshots."""
    # Sample one trial from each condition
    conditions = df['condition'].unique()
    fig, axes = plt.subplots(1, len(conditions), figsize=(15, 5))
    if len(conditions) == 1:
        axes = [axes]
    
    for ax, condition in zip(axes, conditions):
        # Get a random trial from this condition
        condition_df = df[df['condition'] == condition]
        if len(condition_df) == 0:
            continue
        trial = condition_df.sample(n=1).iloc[0]
        
        # Find the corresponding trial image
        trial_id = trial['trial_id']
        trial_image_path = os.path.join(experiment_path, f"trial_{trial_id}.png")
        
        if os.path.exists(trial_image_path):
            # Load and display the image
            img = Image.open(trial_image_path)
            ax.imshow(img)
            # Draw black bounding box
            rect = Rectangle((0, 0), img.width, img.height, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            # Set axis limits and aspect ratio
            ax.set_xlim(0, img.width)
            ax.set_ylim(img.height, 0)
            ax.set_aspect('equal')  # preserve image aspect ratio
            # Add resolution as xlabel
            ax.set_xlabel(f"{img.width}x{img.height}")
            ax.set_title(f"{condition}\nWord: {trial['word']}\nColor: {trial['color']}\nResponse: {trial['response']}")
        else:
            ax.text(0.5, 0.5, f"Image not found for trial {trial_id}",
                   horizontalalignment='center',
                   verticalalignment='center')
            ax.set_title(f"{condition}\nWord: {trial['word']}\nColor: {trial['color']}\nResponse: {trial['response']}")
        # Hide ticks and spines, but keep xlabel
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    figures_path = os.path.join(experiment_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, "example_stimuli.png"))
    plt.close()

def create_meta_example_stimuli_figure(meta_experiment_path: str):
    """Create a figure showing example stimuli from each sub-experiment in the meta-experiment, ordered by variable value."""
    # Get all subdirectories and extract variable info
    subdirs = [d for d in glob(os.path.join(meta_experiment_path, '*')) if os.path.isdir(d)]
    subdir_info = []
    for subdir in subdirs:
        var_name, var_value = extract_variable_from_dir(subdir)
        if var_name is not None:
            subdir_info.append((subdir, var_name, var_value))
    # Sort by variable value
    subdir_info.sort(key=lambda x: x[2])

    # Create figure with subplots for each sub-experiment
    fig, axes = plt.subplots(1, len(subdir_info), figsize=(3*len(subdir_info), 5))
    if len(subdir_info) == 1:
        axes = [axes]
    
    for ax, (subdir, var_name, var_value) in zip(axes, subdir_info):
        # Find a random trial image in this subdirectory
        trial_images = glob(os.path.join(subdir, "trial_*.png"))
        if trial_images:
            # Pick a random trial image
            trial_image_path = random.choice(trial_images)
            img = Image.open(trial_image_path)
            ax.imshow(img)
            # Draw black bounding box
            rect = Rectangle((0, 0), img.width, img.height, linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            # Set axis limits and aspect ratio
            ax.set_xlim(0, img.width)
            ax.set_ylim(img.height, 0)
            ax.set_aspect('equal')  # preserve image aspect ratio
            # Add resolution as xlabel
            ax.set_xlabel(f"{img.width}x{img.height}")
            ax.set_title(f"{var_name} = {var_value}")
        else:
            ax.text(0.5, 0.5, f"No trial images found in {os.path.basename(subdir)}",
                   horizontalalignment='center',
                   verticalalignment='center')
            ax.set_title(f"{var_name} = {var_value}")
        # Hide ticks and spines, but keep xlabel
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    # Save the figure
    figures_path = os.path.join(meta_experiment_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, "meta_example_stimuli.png"))
    plt.close()

def analyze_sequence_alignment(trials: List[Dict]) -> Dict:
    """
    Analyze sequence alignment between AI responses and correct responses across trials.
    
    Args:
        trials: List of trial data dictionaries
        
    Returns:
        Dictionary containing sequence alignment metrics
    """
    # Extract responses and correct answers
    all_responses = []
    all_correct = []
    
    for trial in trials:
        responses = trial["response"]
        words = trial["word"]
        colors = trial["color"]
        
        # Get correct responses based on color
        correct_responses = [get_expected_color_name(color, inverse_color_mapping) for color in colors]
        
        all_responses.extend(responses)
        all_correct.extend(correct_responses)
    
    # Calculate alignment metrics
    total_length = len(all_responses)
    matches = sum(1 for r, c in zip(all_responses, all_correct) if r.lower() == c.lower())
    insertions = sum(1 for r, c in zip(all_responses, all_correct) if c == "-")
    deletions = sum(1 for r, c in zip(all_responses, all_correct) if r == "-")
    substitutions = total_length - matches - insertions - deletions
    
    # Calculate Levenshtein distance for each trial
    trial_distances = []
    for trial in trials:
        responses = trial["response"]
        words = trial["word"]
        colors = trial["color"]
        correct_responses = [get_expected_color_name(color, inverse_color_mapping) for color in colors]
        
        # Calculate distance for each response in the trial
        distances = [levenshtein_distance(r.lower(), c.lower()) for r, c in zip(responses, correct_responses)]
        trial_distances.append(np.mean(distances))
    
    metrics = {
        "match_rate": matches / total_length if total_length > 0 else 0,
        "insertion_rate": insertions / total_length if total_length > 0 else 0,
        "deletion_rate": deletions / total_length if total_length > 0 else 0,
        "substitution_rate": substitutions / total_length if total_length > 0 else 0,
        "total_edit_distance": insertions + deletions + substitutions,
        "mean_levenshtein_distance": np.mean(trial_distances) if trial_distances else 0,
        "std_levenshtein_distance": np.std(trial_distances) if trial_distances else 0
    }
    
    return metrics

def analyze_single_experiment(experiment_path: str) -> pd.DataFrame:
    """Analyze a single experiment and return results DataFrame."""
    experiment_parameters = get_experiment_parameters(experiment_path)
    trials = get_data(experiment_path)
    color_mapping = experiment_parameters["color_mapping"]
    inverse_color_mapping = {v: k for k, v in color_mapping.items() if v is not None}
    
    # Create sequence alignment visualization
    figures_path = os.path.join(experiment_path, 'figures')
    os.makedirs(figures_path, exist_ok=True)
    
    # Initialize Needleman-Wunsch aligner and tokenizer
    aligner = NeedlemanWunsch()
    tokenizer = Tokenizer()
    
    # Get all unique color names to create vocabulary
    all_colors = set()
    for trial in trials:
        responses = [r.lower() for r in trial["response"]]
        correct_responses = [get_expected_color_name(color, inverse_color_mapping).lower() 
                           for color in trial["color"]]
        all_colors.update(responses)
        all_colors.update(correct_responses)
    
    # Create vocabulary mapping
    vocab = {color: idx for idx, color in enumerate(sorted(all_colors))}
    
    # Calculate word-level edit distances
    edit_distances = []
    for trial in trials:
        responses = [r.lower() for r in trial["response"]]
        correct_responses = [get_expected_color_name(color, inverse_color_mapping).lower() 
                           for color in trial["color"]]
        
        # Convert to integer tokens
        response_tokens = [vocab[r] for r in responses]
        correct_tokens = [vocab[c] for c in correct_responses]
        
        # Calculate alignment score
        alignment_score = aligner.get_alignment_score(response_tokens, correct_tokens)
        edit_distances.append(alignment_score)
    
    # Plot edit distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(edit_distances, bins=20)
    plt.title("Distribution of Word-Level Edit Distances")
    plt.xlabel("Edit Distance")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(figures_path, "edit_distance_distribution.png"))
    plt.close()
    
    # analyze number of responses per trial
    number_of_responses_per_trial = [len(t["response"]) for t in trials]
    expected_number_of_responses_per_trial = experiment_parameters["number_of_words_per_trial"]
    print(f"Expected number of responses per trial: {expected_number_of_responses_per_trial}, but got {np.mean(number_of_responses_per_trial)} +- {np.std(number_of_responses_per_trial)} (min: {np.min(number_of_responses_per_trial)}, max: {np.max(number_of_responses_per_trial)})")
    
    # distribution of number of responses per trial
    fig = px.histogram(number_of_responses_per_trial)
    fig.update_xaxes(title="Number of responses per trial")
    fig.update_yaxes(title="Count")
    fig.update_layout(showlegend=False)
    fig.write_image(os.path.join(figures_path, "number_of_responses_per_trial_histogram.png"))
    
    results = []
    for trial in trials:
        for word_index, (word, color, response) in enumerate(zip(trial["word"], trial["color"], trial["response"])):
            response = response.lower().strip()
            condition = get_condition(word, color, color_mapping)
            correct_word = is_correct_word(word, response)
            color_name = get_expected_color_name(color, inverse_color_mapping)
            correct_color = is_correct_color(color_name, response, legit_color_confusion)
            results.append({
                "trial_id": trial["trial_id"],
                "word_index": word_index,
                "word": word,
                "color": color,
                "color_name": color_name,
                "response": response,
                "condition": condition,
                "correct_color": correct_color,
                "correct_word": correct_word,
                "is_neutral": color_mapping[word] is None
            })
    
    df = pd.DataFrame(results)
    
    # Create example stimuli figure
    create_example_stimuli_figure(df, experiment_path)
    
    # probability to response with a name which is not a color
    df['response_color_name'] = df['response'].map(lambda response: get_response_color_name(response, legit_color_confusion))
    df['valid_color'] = df['response_color_name'].isin([c.lower() for c in inverse_color_mapping.values()])
    print(f"Probability to response with a name which is color: {np.mean(df['valid_color'])}")      
    # examples for non-color responses
    logger.info("Non valid color responses")
    df_non_valid_color = df[~df['valid_color']]
    logger.info(df_non_valid_color.groupby(['color','word','response']).size().sort_values(ascending=False))
    # totally wrong responses
    logger.info("Totally wrong responses")
    df['totally_wrong'] = (~df['correct_color'] & ~df['correct_word'])
    logger.info(df[df['totally_wrong']]['response'].value_counts())
    df_totally_wrong = df[df['totally_wrong']]
    logger.info(df_totally_wrong.groupby(['color','word','response']).size().sort_values(ascending=False))

    # Calculate mean and standard error for each condition
    summary = df.groupby('condition')['correct_color'].agg(['mean', 'count', 'std']).reset_index()
    summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
    
    # Create bar plot for accuracy by condition
    fig = px.bar(summary, 
                 x="condition", 
                 y="mean", 
                 color="condition",
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
    fig = px.bar(summary_errors, 
                 x="condition", 
                 y="mean", 
                 color="condition",
                 error_y="stderr",
                 labels={"mean": "Accuracy"})
    fig.update_xaxes(title="Condition")
    fig.update_yaxes(title="Naming word accuracy")
    fig.write_image(os.path.join(figures_path, "naming_word_accuracy_by_condition_barplot.png"))
    
    num_of_response_words_mean = np.mean(number_of_responses_per_trial)
    num_of_response_words_std = np.std(number_of_responses_per_trial)
    num_of_response_stats = {'mean': num_of_response_words_mean, 'std': num_of_response_words_std}
#    num_of_response_stats.update(alignment_metrics)
    return df, df_errors, num_of_response_stats

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
    # Create example stimuli figure first
    create_meta_example_stimuli_figure(meta_experiment_path)
    
    all_results = []
    variable_name = None
    variable_values = []
    
    # Add sequence alignment analysis for meta-experiment
    all_alignment_metrics = []
    
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
        df, df_errors, num_of_response_stats = analyze_single_experiment(subdir)
        df[var_name] = var_value
        all_results.append(df)
        variable_name = var_name
        variable_values.append(var_value)
        
        # Add sequence alignment analysis for each sub-experiment
#        alignment_metrics = analyze_sequence_alignment(get_data(subdir))
#        alignment_metrics[var_name] = var_value
#        all_alignment_metrics.append(alignment_metrics)
    
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
    
    # Create alignment metrics visualization
    if all_alignment_metrics:
        alignment_df = pd.DataFrame(all_alignment_metrics)
        variable_name = next(iter(set(alignment_df.columns) - set(['match_rate', 'insertion_rate', 'deletion_rate', 
                                                                 'substitution_rate', 'total_edit_distance',
                                                                 'mean_levenshtein_distance', 'std_levenshtein_distance'])))
        
        # Plot alignment metrics by variable
        fig = px.line(alignment_df, x=variable_name, y=['match_rate', 'insertion_rate', 'deletion_rate', 'substitution_rate'],
                     labels={'value': 'Rate', 'variable': 'Metric'},
                     title=f'Sequence Alignment Metrics by {variable_name.replace("_", " ").title()}')
        fig.write_image(os.path.join(figures_path, f"alignment_metrics_by_{variable_name}.png"))
        
        # Plot Levenshtein distance by variable
        fig = px.line(alignment_df, x=variable_name, y=['mean_levenshtein_distance'],
                     error_y='std_levenshtein_distance',
                     labels={'value': 'Distance', 'variable': 'Metric'},
                     title=f'Mean Levenshtein Distance by {variable_name.replace("_", " ").title()}')
        fig.write_image(os.path.join(figures_path, f"levenshtein_distance_by_{variable_name}.png"))
    
    return combined_df, summary

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze multiword experiment results')
    parser.add_argument('experiment_path', type=str, help='Path to the experiment results folder')
    parser.add_argument('--meta', action='store_true', help='Whether this is a meta-experiment (contains multiple sub-experiments)')
    args = parser.parse_args()
    
    if args.meta:
        # For meta-experiment analysis
        logger.info(f"Analyzing meta-experiment: {args.experiment_path}")
        combined_df, summary = analyze_meta_experiment(args.experiment_path)
        
        # Print summary statistics
        print("\nMeta-experiment summary:")
        print(summary.to_string())
    else:
        # For single experiment analysis
        logger.info(f"Analyzing single experiment: {args.experiment_path}")
        df, df_errors, num_of_response_stats = analyze_single_experiment(args.experiment_path)
        
        # Calculate summary statistics with error bars
        summary = df.groupby('condition')['correct_color'].agg(['mean', 'count', 'std']).reset_index()
        summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
        
        # Create bar plot with error bars
        fig = px.bar(summary, x="condition", y="mean", color="condition",
                     error_y="stderr",
                     labels={"mean": "Accuracy", "condition": "Condition"},
                     title="Accuracy by Condition")
        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False
        )
        
        # Save the plot
        figures_path = os.path.join(args.experiment_path, 'figures')
        os.makedirs(figures_path, exist_ok=True)
        fig.write_image(os.path.join(figures_path, "accuracy_by_condition_with_error_bars.png"))
        
        # Print summary statistics
        print("\nExperiment summary:")
        print(summary.to_string())
        
        # Additional analysis for single experiment
        print("\nResponse distribution:")
        print(df['response'].value_counts().head(10))
        
        print("\nError analysis:")
        error_df = df[~df['correct_color']]
        print("\nMost common incorrect responses:")
        print(error_df['response'].value_counts().head(10))
        
        print("\nError patterns by condition:")
        error_patterns = error_df.groupby('condition')['response'].value_counts().groupby('condition').head(3)
        print(error_patterns)

if __name__ == "__main__":
    main()