import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
from typing import Optional, Union, List, Tuple


def visualize_following_distribution(
    csv_path: str,
    bin_size: int = 10,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Visualize the distribution of following_list and following_agentid_list in a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        bin_size (int): Size of bins for histogram (e.g., 10 means 0-9, 10-19, etc.)
        save_path (Optional[str]): Path to save the figure, if None, the figure won't be saved
        show_plot (bool): Whether to display the plot
        figsize (Tuple[int, int]): Figure size (width, height)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Process following_list
    following_counts = []
    for item in df['following_list']:
        try:
            # Convert string representation of list to actual list
            following = ast.literal_eval(item)
            following_counts.extend(following)
        except (ValueError, SyntaxError):
            print(f"Error parsing following_list: {item}")
    
    # Process following_agentid_list
    agentid_counts = []
    for item in df['following_agentid_list']:
        try:
            # Convert string representation of list to actual list
            agentids = ast.literal_eval(item)
            agentid_counts.extend(agentids)
        except (ValueError, SyntaxError):
            print(f"Error parsing following_agentid_list: {item}")
    
    # Calculate bin edges based on the maximum value and bin_size
    max_following = max(following_counts) if following_counts else 0
    max_agentid = max(agentid_counts) if agentid_counts else 0
    max_value = max(max_following, max_agentid)
    
    # Create bin edges from 0 to max_value with step bin_size
    bin_edges = np.arange(0, max_value + bin_size + 1, bin_size)
    
    # Plot histograms
    axes[0].hist(following_counts, bins=bin_edges, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribution of following_list Values')
    axes[0].set_xlabel('User ID Range')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].hist(agentid_counts, bins=bin_edges, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Distribution of following_agentid_list Values')
    axes[1].set_xlabel('Agent ID Range')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add bin range labels to x-axis
    for ax in axes:
        ax.set_xticks(bin_edges[:-1] + bin_size/2)
        ax.set_xticklabels([f'{int(edge)}-{int(edge+bin_size-1)}' for edge in bin_edges[:-1]])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    # Show the plot if show_plot is True
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    visualize_following_distribution(
        csv_path=r"data\our_twitter_sim\test_1000_good_bad_random_bernoulli_wlx.csv",
        bin_size=50,
        show_plot=True
    )
