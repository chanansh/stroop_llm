import pandas as pd
import numpy as np

def analyze_stroop_responses(df):
    # Set condition based on word and color match
    df['condition'] = np.where(df['word'].str.lower() == df['color'].str.lower(), 'congruent', 'incongruent')
    df.loc[df['word'] == 'XXXX', 'condition'] = 'neutral'
    
    # Initialize correct column
    df['correct'] = False
    
    # Extract response word and color from prompt
    df['response_word'] = df['prompt'].str.extract('word is (\\w+)')
    df['response_color'] = df['prompt'].str.extract('(\\w+) color')
    
    # Calculate correctness for each experiment type
    df.loc[df['experiment'] == 'meaning_only', 'correct'] = (
        (df['prompt'].str.contains('blue') & (df['word'].str.lower() == 'blue')) |
        (df['prompt'].str.contains('red') & (df['word'].str.lower() == 'red'))
    )
    
    df.loc[df['experiment'] == 'color_only', 'correct'] = (
        (df['prompt'].str.contains('blue') & (df['color'].str.lower() == 'blue')) |
        (df['prompt'].str.contains('red') & (df['color'].str.lower() == 'red'))
    )
    
    df.loc[df['experiment'] == 'combined', 'correct'] = (
        (df['response_word'].str.lower() == df['word'].str.lower()) &
        (df['response_color'].str.lower() == df['color'].str.lower())
    )
    
    return df

# Example usage:
# df = pd.read_csv('your_data.csv')
# df = analyze_stroop_responses(df) 