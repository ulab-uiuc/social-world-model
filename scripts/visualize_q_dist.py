import os
import json
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def read_json_file(file_path):
    """
    Reads a single JSON lines file and returns its data.
    """
    try:
        with jsonlines.open(file_path, 'r') as file:
            return [data for data in file]
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def read_json_files(directory_path):
    """
    Reads all JSON lines files in parallel using ThreadPoolExecutor.
    """
    directory = Path(directory_path)
    json_files = list(directory.glob('*.json*'))
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_json_file, json_files))
    
    return [result for result in results if result]

def process_domain_data(domain):
    """
    Process data for a single domain.
    """
    directory = f'../cache/cache_{domain}_basicpriorreasoner'
    return read_json_files(directory)

def create_styled_plot(df, highlight_step=20):
    """
    Creates a styled plot with shadow effect and optimized appearance using seaborn.
    """
    # Set seaborn style
    sns.set_style("whitegrid", {'axes.grid': True,
                               'grid.color': '.8',
                               'grid.linestyle': '--',
                               'grid.alpha': 0.5})
    sns.set_context("poster", font_scale=1.2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10), dpi=100)
    
    # Create shadow effect
    x = df.index
    y = df['Average']
    
    # Color palette
    palette = sns.color_palette("deep")
    main_color = palette[0]
    
    # Add gradient shadow under the curve
    # Base shadow - only under the actual curve
    mask = ~np.isnan(y)  # Create mask for non-NaN values
    ax.fill_between(x[mask], y[mask], 
                   color=main_color, alpha=0.15,
                   zorder=2)
    
    # Plot main line with seaborn color
    sns.lineplot(data=df, x=df.index, y='Average', 
                color=main_color, linewidth=4, 
                label='Importance Score',  # Shortened label
                zorder=3, ax=ax)
    
    # Highlight points
    highlight_indices = range(0, len(df), highlight_step)
    highlight_values = df['Average'].iloc[highlight_indices]
    
    # Add scatter points with seaborn styling
    ax.scatter(highlight_indices, highlight_values, 
              color=main_color, s=400, zorder=4,
              alpha=0.7)
    
    # Annotations with seaborn font scaling
    for idx, value in zip(highlight_indices, highlight_values):
        if np.isfinite(value):
            ax.text(idx+10, value, f"{value:.3f}", 
                   fontsize=35, color='#2f2f2f', 
                   ha='right', va='bottom',
                   fontweight='bold')
    
    # Styling with seaborn parameters
    ax.set_xlabel('Ranking Index', fontsize=50, labelpad=15)
    ax.set_ylabel('Importance Score', fontsize=50, labelpad=15)
    ax.tick_params(axis='both', labelsize=45, colors='#2f2f2f')

    
    # Set limits with padding
    y_padding = (df['Average'].max() - df['Average'].min()) * 0.05
    ax.set_ylim(df['Average'].min() - y_padding, 
                df['Average'].max() + y_padding)
    
    # Legend styling - reduced size
    ax.legend(loc='upper right', fontsize=35, frameon=True,
             facecolor='white', framealpha=0.9,
             edgecolor='none')
    
    # Additional seaborn styling
    sns.despine(left=False, bottom=False)
    
    # Refine grid
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1,
           color='gray', which='major')
    
    return fig, ax

def main():
    domains = ['crypto', 'other', 'election', 'politics', 'sports']
    
    # Process domains in parallel
    with ThreadPoolExecutor() as executor:
        domain_results = list(executor.map(process_domain_data, domains))
    
    # Flatten results
    overall_dataset = [item for sublist in domain_results for item in sublist]
    
    # Filter and process data
    filtered_data = [
        [d.get('score', 0) for d in data]
        for data in overall_dataset
        if data and 'score' in data[0] and data[0]['score'] > 0
    ]
    
    print(f"Total filtered distributions: {len(filtered_data)}")
    
    if not filtered_data:
        print("No distributions to plot after filtering.")
        return
    
    # Create DataFrame
    max_length = max(len(d) for d in filtered_data)
    padded_data = [d + [np.nan] * (max_length - len(d)) for d in filtered_data]
    df = pd.DataFrame(padded_data).transpose()
    df.columns = [f'Distribution {i+1}' for i in range(len(filtered_data))]
    df.index.name = 'Index'
    df['Average'] = df.mean(axis=1, skipna=True)
    
    # Create and save plot
    fig, ax = create_styled_plot(df)
    plt.tight_layout()
    
    # Save with high quality settings
    plt.savefig('average_value_trend_distributions.pdf', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor='white',
                edgecolor='none')
    plt.show()

if __name__ == "__main__":
    main()