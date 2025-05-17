import plotly.express as px
import plotly.subplots as make_subplots
import plotly.graph_objects as go
import os
from PIL import Image
import numpy as np

# Define the grid
words = ["BLUE", "RED", "XXXX"]
colors = [("blue", "0_0_255"), ("red", "255_0_0")]

# Create subplot figure
fig = make_subplots.make_subplots(
    rows=2, cols=3,
    subplot_titles=words,
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)

# Add images to subplots
for row, (color_name, rgb) in enumerate(colors):
    for col, word in enumerate(words):
        fname = f"{word}_{rgb}.png"
        fpath = os.path.join("../stimuli_images", fname)
        
        if os.path.exists(fpath):
            img = Image.open(fpath)
            # Convert to RGB mode if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image with high-quality resampling
            img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
            img_array = np.array(img)
            
            fig.add_trace(
                go.Image(z=img_array),
                row=row+1, col=col+1
            )

# Update layout
fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showticklabels=False, showgrid=False)

# Add y-axis labels
for i, (color_name, _) in enumerate(colors):
    fig.add_annotation(
        x=-0.1,
        y=0.75-i*0.5,
        text=color_name.capitalize(),
        textangle=-90,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=14)
    )

# Add x-axis label
fig.add_annotation(
    x=0.5,
    y=-0.15,
    text='Word',
    xref='paper',
    yref='paper',
    showarrow=False,
    font=dict(size=14)
)

# Update layout with higher resolution settings
fig.update_layout(
    width=1200,  # Doubled the width
    height=600,  # Doubled the height
    showlegend=False,
    margin=dict(l=80, r=20, t=40, b=60)
)

# Save with higher DPI and quality settings
fig.write_image('../results/figures/stimuli.png', scale=4, engine='kaleido')
fig.show() 