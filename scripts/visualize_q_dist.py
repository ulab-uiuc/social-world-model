import os
import json
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

def read_json_files(directory_path):
    """
    Reads all JSON lines files in the specified directory and returns a list of lists of scores.

    :param directory_path: Path to the directory containing JSON files.
    :return: List of lists containing scores.
    """
    json_data = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a JSON lines file
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            file_path = os.path.join(directory_path, filename)
            try:
                with jsonlines.open(file_path, 'r') as file:
                    dataset = [data for data in file]
                    json_data.append(dataset)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return json_data

def main():
    # Define the domains and corresponding directories
    domains = ['crypto', 'other', 'election', 'politics', 'sports']
    overall_dataset = []
    
    for domain in domains:
        directory = f'../cache/cache_{domain}_basicpriorreasoner'
        data = read_json_files(directory)
        overall_dataset.extend(data)
        print(f"Read {len(data)} JSON files for domain '{domain}'.")
    
    # Filter datasets where the first 'score' > 0 and extract scores
    filtered_overall_dataset = []
    for data in overall_dataset:
        if data and 'score' in data[0] and data[0]['score'] > 0:
            scores = [d.get('score', 0) for d in data]
            filtered_overall_dataset.append(scores)
    
    print(f"Total filtered distributions: {len(filtered_overall_dataset)}")
    
    # Check if there are any distributions to process
    if not filtered_overall_dataset:
        print("No distributions to plot after filtering.")
        return
    
    # Determine the maximum length of distributions
    max_length = max(len(d) for d in filtered_overall_dataset)
    
    # Pad shorter distributions with NaN for alignment
    padded_data = [d + [np.nan]*(max_length - len(d)) for d in filtered_overall_dataset]
    
    # Create a DataFrame where each column represents a distribution
    df = pd.DataFrame(padded_data).transpose()
    
    # Rename columns for clarity
    df.columns = [f'Distribution {i+1}' for i in range(len(filtered_overall_dataset))]
    
    # Assign index name if applicable
    df.index.name = 'Index'
    
    # Calculate the average trend across all distributions
    df['Average'] = df.mean(axis=1, skipna=True)

    import pdb; pdb.set_trace()
    
    # Set font to Times New Roman if available, else use DejaVu Serif
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    if 'Times New Roman' in available_fonts:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    else:
        print("Times New Roman not found. Using DejaVu Serif as fallback.")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # Update font sizes and other plot parameters
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 26
    })

    # Create the plot
    plt.figure(figsize=(15, 9))
    ax = plt.gca()
    
    # Plot the average trend without markers
    plt.plot(
        df.index, 
        df['Average'], 
        color='#d62728',           # Distinct color for the average trend
        linewidth=3,               # Thicker line for prominence
        label='Average Trend'      # Label for the legend
    )
    
    # Set plot titles and labels
    plt.title('Average Value Trend Across Distributions', fontsize=30)
    plt.xlabel('Index', fontsize=30)  # Replace 'Index' with specific category names if applicable
    plt.ylabel('Average Score', fontsize=30)
    
    # Customize x-axis ticks
    plt.xticks(fontsize=30)
    
    # Customize y-axis ticks
    plt.yticks(fontsize=30)
    
    # Set y-axis limits based on data with padding
    plt.ylim(df['Average'].min() - 0.05, df['Average'].max() + 0.05)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend with customized font sizes
    legend = plt.legend(loc='upper right', title='Legend', title_fontsize=30)
    plt.setp(legend.get_title(), fontsize='26')
    plt.setp(legend.get_texts(), fontsize='26')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the plot as a high-resolution PDF
    plt.savefig('average_value_trend_distributions.pdf', dpi=300)
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
