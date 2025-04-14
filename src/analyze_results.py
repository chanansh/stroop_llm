import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare the CLIP analysis results."""
    df = pd.read_csv("results/clip_analysis.csv")
    
    # Add congruency column
    df['congruent'] = df.apply(lambda row: 
        'congruent' if (
            (row['word'] == 'RED' and row['color'] == 'red') or 
            (row['word'] == 'BLUE' and row['color'] == 'blue')
        ) else 'incongruent' if row['word'] != 'XXXX' else 'control',
        axis=1
    )
    
    return df

def plot_heatmap(df):
    """Create a heatmap of probabilities for different conditions."""
    plt.figure(figsize=(15, 10))
    
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        values='probability',
        index=['word', 'color'],
        columns='class_name',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0.5)
    plt.title('CLIP Probabilities Across Conditions')
    plt.tight_layout()
    plt.savefig('results/heatmap.png')
    plt.close()

def plot_congruency_effects(df):
    """Plot the effects of congruency on CLIP predictions."""
    plt.figure(figsize=(12, 6))
    
    # Filter for combined class set and correct predictions
    combined_df = df[
        (df['class_set'] == 'combined') & 
        (df['word'] != 'XXXX') &
        (df.apply(lambda x: x['class_name'] == f"the word is {x['word'].lower()} in {x['color']} color", axis=1))
    ]
    
    # Create grouped bar plot
    sns.barplot(
        data=combined_df,
        x='word',
        y='probability',
        hue='congruent',
        errorbar=None
    )
    
    plt.title('Congruency Effects in CLIP Predictions')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig('results/congruency_effects.png')
    plt.close()

def plot_accuracy_by_task(df):
    """Plot accuracy for different recognition tasks."""
    plt.figure(figsize=(12, 6))
    
    # Calculate accuracy for each task type
    task_accuracy = []
    
    # Word recognition accuracy (when the word matches the class)
    word_acc = df[df['class_set'] == 'word_only'].apply(
        lambda x: x['probability'] if x['class_name'] == f"the word is {x['word'].lower()}" else 0,
        axis=1
    ).mean()
    task_accuracy.append(('Word Recognition', word_acc))
    
    # Color recognition accuracy (when the color matches the class)
    color_acc = df[df['class_set'] == 'color_only'].apply(
        lambda x: x['probability'] if x['class_name'] == f"the color is {x['color']}" else 0,
        axis=1
    ).mean()
    task_accuracy.append(('Color Recognition', color_acc))
    
    # Combined task accuracy (congruent trials)
    cong_acc = df[
        (df['class_set'] == 'combined') & 
        (df['congruent'] == 'congruent')
    ].apply(
        lambda x: x['probability'] if x['class_name'] == f"the word is {x['word'].lower()} in {x['color']} color" else 0,
        axis=1
    ).mean()
    task_accuracy.append(('Combined (Congruent)', cong_acc))
    
    # Combined task accuracy (incongruent trials)
    incong_acc = df[
        (df['class_set'] == 'combined') & 
        (df['congruent'] == 'incongruent')
    ].apply(
        lambda x: x['probability'] if x['class_name'] == f"the word is {x['word'].lower()} in {x['color']} color" else 0,
        axis=1
    ).mean()
    task_accuracy.append(('Combined (Incongruent)', incong_acc))
    
    # Create bar plot
    acc_df = pd.DataFrame(task_accuracy, columns=['Task', 'Accuracy'])
    sns.barplot(data=acc_df, x='Task', y='Accuracy', errorbar=None)
    plt.title('CLIP Accuracy by Task Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/accuracy_by_task.png')
    plt.close()

def analyze_stroop_effect(df):
    """Analyze and print statistics about the Stroop effect."""
    # Filter for combined trials
    combined_df = df[df['class_set'] == 'combined']
    
    # Calculate mean probabilities for congruent vs incongruent trials
    cong_mean = combined_df[combined_df['congruent'] == 'congruent']['probability'].mean()
    incong_mean = combined_df[combined_df['congruent'] == 'incongruent']['probability'].mean()
    
    # Calculate interference effect
    interference = cong_mean - incong_mean
    
    print("\nStroop Effect Analysis:")
    print(f"Congruent trials mean probability: {cong_mean:.3f}")
    print(f"Incongruent trials mean probability: {incong_mean:.3f}")
    print(f"Interference effect: {interference:.3f}")
    
    # Analyze control condition
    control_mean = combined_df[combined_df['congruent'] == 'control']['probability'].mean()
    print(f"Control condition mean probability: {control_mean:.3f}")

def main():
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create visualizations
    plot_heatmap(df)
    plot_congruency_effects(df)
    plot_accuracy_by_task(df)
    
    # Analyze Stroop effect
    analyze_stroop_effect(df)
    
    print("\nAnalysis complete. Visualizations saved in results/")

if __name__ == "__main__":
    main() 